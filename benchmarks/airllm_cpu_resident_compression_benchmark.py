#!/usr/bin/env python3
"""Benchmark AirLLM compression with every compressed shard resident in CPU.

The preparation phase may split and compress an ordinary BF16 checkpoint.
After AirLLM is initialized, this benchmark loads every compressed per-layer
shard into a process-owned CPU cache. The timed inference window therefore
contains CPU->GPU transfer, GPU decompression, module installation, compute,
and release, but no intentional SSD model reads.
"""

import argparse
import ctypes
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

import psutil
import torch
from airllm import AutoModel
from airllm.persist import ModelPersister
from airllm.utils import uncompress_layer_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer-shards-root", type=Path, required=True)
    parser.add_argument("--compression", choices=("8bit", "4bit"), default="8bit")
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--warmup-decode", type=int, default=1)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--profile-repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pin-cpu-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def process_snapshot():
    process = psutil.Process()
    memory = process.memory_info()
    io = process.io_counters()
    status = {}
    with Path("/proc/self/status").open("r", encoding="utf-8") as source:
        for line in source:
            if line.startswith(("VmLck:", "RssAnon:", "RssFile:")):
                name, value, unit = line.split()
                if unit != "kB":
                    raise RuntimeError("unexpected /proc status unit")
                status[name.rstrip(":")] = int(value) * 1024
    return {
        "rss_bytes": memory.rss,
        "vms_bytes": memory.vms,
        "read_bytes": io.read_bytes,
        "write_bytes": io.write_bytes,
        "vm_locked_bytes": status.get("VmLck", 0),
        "rss_anon_bytes": status.get("RssAnon", 0),
        "rss_file_bytes": status.get("RssFile", 0),
        "system_available_bytes": psutil.virtual_memory().available,
    }


def tensor_bytes(state_dict):
    return sum(
        value.numel() * value.element_size()
        for value in state_dict.values()
        if isinstance(value, torch.Tensor)
    )


def pinned_tensor_bytes(state_dict):
    return sum(
        value.numel() * value.element_size()
        for value in state_dict.values()
        if isinstance(value, torch.Tensor) and value.is_pinned()
    )


def distribution(values):
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": min(ordered),
        "median": statistics.median(ordered),
        "max": max(ordered),
        "mean": statistics.mean(ordered),
    }


def materialize_state_dict(state_dict, pin_memory):
    """Detach safetensors mmap storage into process-owned CPU DRAM."""
    materialized = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            materialized[key] = (
                value.pin_memory() if pin_memory else value.clone()
            )
        else:
            materialized[key] = value
    return materialized


def main():
    args = parse_args()
    if args.warmup_decode < 0:
        raise SystemExit("--warmup-decode cannot be negative")
    if args.decode_repeats < 1 or args.profile_repeats < 1:
        raise SystemExit("decode/profile repeats must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.layer_shards_root.mkdir(parents=True, exist_ok=True)
    before_init = process_snapshot()
    init_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(args.checkpoint),
        device=args.device,
        dtype=torch.bfloat16,
        layer_shards_saving_path=str(args.layer_shards_root),
        profiling_mode=False,
        compression=args.compression,
        prefetching=False,
        delete_original=False,
    )
    initialization_seconds = time.perf_counter() - init_started
    after_init = process_snapshot()

    # Load the *compressed* state dicts directly. Calling AirLLM's normal
    # load_layer_to_cpu here would already copy/decompress them on the GPU.
    persister = ModelPersister.get_model_persister()
    cache_started = time.perf_counter()
    cache_io_start = process_snapshot()
    cpu_cache = {}
    for layer_name in model.layer_names:
        state_dict = persister.load_model(layer_name, model.checkpoint_path)
        cpu_cache[layer_name] = materialize_state_dict(
            state_dict,
            pin_memory=args.pin_cpu_cache,
        )
        del state_dict
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    cache_io_end = process_snapshot()
    cpu_cache_seconds = time.perf_counter() - cache_started
    cpu_cache_bytes = sum(tensor_bytes(item) for item in cpu_cache.values())
    cpu_cache_pinned_bytes = sum(
        pinned_tensor_bytes(item) for item in cpu_cache.values()
    )
    after_cpu_cache = process_snapshot()

    layer_profile = {
        "calls": 0,
        "wall_ms": 0.0,
        "compressed_h2d_bytes": 0,
    }
    profile_enabled = False

    def cpu_resident_load(layer_name):
        started = time.perf_counter()
        compressed = cpu_cache[layer_name]
        result = uncompress_layer_state_dict(compressed)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if profile_enabled:
            layer_profile["calls"] += 1
            layer_profile["wall_ms"] += elapsed_ms
            layer_profile["compressed_h2d_bytes"] += tensor_bytes(compressed)
        return result

    model.load_layer_to_cpu = cpu_resident_load
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    encoded = model.tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    past_key_values = None
    generated = []

    def step(current_ids, current_mask):
        nonlocal past_key_values
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start_event.record(torch.cuda.current_stream(device))
        output = model(
            input_ids=current_ids,
            attention_mask=current_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        end_event.record(torch.cuda.current_stream(device))
        end_event.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        gpu_ms = start_event.elapsed_time(end_event)
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token.detach().cpu())
        next_mask = torch.cat(
            (
                current_mask,
                torch.ones(
                    (current_mask.shape[0], 1),
                    dtype=current_mask.dtype,
                    device=device,
                ),
            ),
            dim=1,
        )
        return next_token, next_mask, wall_ms, gpu_ms

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        current, attention_mask, prefill_wall_ms, prefill_gpu_ms = step(
            input_ids,
            attention_mask,
        )
        for _ in range(args.warmup_decode):
            current, attention_mask, _, _ = step(current, attention_mask)

        measured_io_start = process_snapshot()
        decode_wall_ms = []
        decode_gpu_ms = []
        for _ in range(args.decode_repeats):
            current, attention_mask, wall_ms, gpu_ms = step(
                current,
                attention_mask,
            )
            decode_wall_ms.append(wall_ms)
            decode_gpu_ms.append(gpu_ms)
        measured_io_end = process_snapshot()

        profile_enabled = True
        profile_rows = []
        for _ in range(args.profile_repeats):
            before = dict(layer_profile)
            current, attention_mask, wall_ms, gpu_ms = step(
                current,
                attention_mask,
            )
            profile_rows.append(
                {
                    "wall_ms": wall_ms,
                    "gpu_ms": gpu_ms,
                    "layer_load_decompress_calls": (
                        layer_profile["calls"] - before["calls"]
                    ),
                    "layer_load_decompress_wall_ms": (
                        layer_profile["wall_ms"] - before["wall_ms"]
                    ),
                    "compressed_h2d_bytes": (
                        layer_profile["compressed_h2d_bytes"]
                        - before["compressed_h2d_bytes"]
                    ),
                }
            )
        profile_enabled = False

    generated_ids = torch.cat(generated, dim=-1)
    report = {
        "schema_version": 1,
        "implementation": "AirLLM CPU-resident compressed cache",
        "airllm_version": importlib.metadata.version("airllm"),
        "checkpoint": str(args.checkpoint),
        "compressed_shards_path": str(model.checkpoint_path),
        "compression": model.compression,
        "runtime_dtype": str(model.running_dtype),
        "prefetching": bool(model.prefetching),
        "cpu_cache_mode": (
            "pinned" if args.pin_cpu_cache else "pageable"
        ),
        "initialization_seconds_excluded": initialization_seconds,
        "cpu_cache_load_seconds_excluded": cpu_cache_seconds,
        "cpu_cache_bytes": cpu_cache_bytes,
        "cpu_cache_pinned_bytes": cpu_cache_pinned_bytes,
        "cpu_cache_io_delta": {
            "read_bytes": (
                cache_io_end["read_bytes"] - cache_io_start["read_bytes"]
            ),
            "write_bytes": (
                cache_io_end["write_bytes"] - cache_io_start["write_bytes"]
            ),
        },
        "prompt": args.prompt,
        "prompt_tokens": int(input_ids.numel()),
        "streamed_units": len(model._streamed_indices),
        "prefill_wall_ms": prefill_wall_ms,
        "prefill_gpu_ms": prefill_gpu_ms,
        "warmup_decode": args.warmup_decode,
        "decode_repeats": args.decode_repeats,
        "decode_wall_ms": decode_wall_ms,
        "decode_gpu_ms": decode_gpu_ms,
        "decode_wall_distribution_ms": distribution(decode_wall_ms),
        "decode_gpu_distribution_ms": distribution(decode_gpu_ms),
        "profile_repeats": args.profile_repeats,
        "profile_rows": profile_rows,
        "measured_decode_io_delta": {
            "read_bytes": (
                measured_io_end["read_bytes"]
                - measured_io_start["read_bytes"]
            ),
            "write_bytes": (
                measured_io_end["write_bytes"]
                - measured_io_start["write_bytes"]
            ),
        },
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
            device
        ),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "generated_token_ids": generated_ids.tolist(),
        "generated_text": model.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ),
        "before_init": before_init,
        "after_init": after_init,
        "after_cpu_cache": after_cpu_cache,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "bitsandbytes_version": importlib.metadata.version("bitsandbytes"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
