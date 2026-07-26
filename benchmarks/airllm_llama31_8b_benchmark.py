#!/usr/bin/env python3
"""Benchmark AirLLM on the same real Llama-3.1-8B decode protocol as Cascade."""

import argparse
import importlib.metadata
import json
import os
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch
from airllm import AutoModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer-shards-root", type=Path, required=True)
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--warmup-decode", type=int, default=1)
    parser.add_argument("--decode-repeats", type=int, default=7)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def process_snapshot():
    process = psutil.Process()
    memory = process.memory_info()
    io = process.io_counters()
    status_values = {}
    with Path("/proc/self/status").open("r", encoding="utf-8") as source:
        for line in source:
            if line.startswith(("VmLck:", "RssAnon:", "RssFile:")):
                name, value, unit = line.split()
                if unit != "kB":
                    raise RuntimeError("unexpected /proc status unit")
                status_values[name.rstrip(":")] = int(value) * 1024
    return {
        "rss_bytes": memory.rss,
        "vms_bytes": memory.vms,
        "read_bytes": io.read_bytes,
        "write_bytes": io.write_bytes,
        "vm_locked_bytes": status_values.get("VmLck", 0),
        "rss_anon_bytes": status_values.get("RssAnon", 0),
        "rss_file_bytes": status_values.get("RssFile", 0),
    }


def distribution(values):
    ordered = sorted(values)
    count = len(ordered)

    def percentile(fraction):
        if count == 1:
            return ordered[0]
        position = fraction * (count - 1)
        lower = int(position)
        upper = min(lower + 1, count - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": count,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
        "min": min(values),
        "max": max(values),
    }


def tree_stats(path):
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def main():
    args = parse_args()
    if args.warmup_decode < 0:
        raise SystemExit("--warmup-decode cannot be negative")
    if args.decode_repeats < 1 or args.profile_repeats < 0:
        raise SystemExit("repeat counts are invalid")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.layer_shards_root.mkdir(parents=True, exist_ok=True)
    split_path = args.layer_shards_root / "splitted_model"
    split_before = tree_stats(split_path)
    before_init = process_snapshot()

    init_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(args.checkpoint),
        device=args.device,
        dtype=torch.bfloat16,
        layer_shards_saving_path=str(args.layer_shards_root),
        profiling_mode=False,
        compression=None,
        prefetching=True,
        delete_original=False,
    )
    init_seconds = time.perf_counter() - init_started
    after_init = process_snapshot()
    split_after = tree_stats(split_path)

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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    generated = []
    past_key_values = None

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

    with torch.inference_mode():
        current, attention_mask, prefill_wall_ms, prefill_gpu_ms = step(
            input_ids,
            attention_mask,
        )
        for _ in range(args.warmup_decode):
            current, attention_mask, _, _ = step(current, attention_mask)

        measured_start = process_snapshot()
        decode_wall_ms = []
        decode_gpu_ms = []
        for _ in range(args.decode_repeats):
            current, attention_mask, wall_ms, gpu_ms = step(
                current,
                attention_mask,
            )
            decode_wall_ms.append(wall_ms)
            decode_gpu_ms.append(gpu_ms)
        measured_end = process_snapshot()

        profile_rows = []
        original_load = model.load_layer_to_cpu
        original_move = model.move_layer_to_device
        profile_lock = threading.Lock()
        profile_state = {
            "load_calls": 0,
            "load_wall_ms": 0.0,
            "move_calls": 0,
            "move_wall_ms": 0.0,
            "move_gpu_ms": 0.0,
            "move_bytes": 0,
            "move_pinned_bytes": 0,
        }

        def profiled_load(layer_name):
            started = time.perf_counter()
            state_dict = original_load(layer_name)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with profile_lock:
                profile_state["load_calls"] += 1
                profile_state["load_wall_ms"] += elapsed_ms
            return state_dict

        def profiled_move(state_dict):
            values = list(state_dict.values())
            payload_bytes = sum(
                value.numel() * value.element_size()
                for value in values
                if isinstance(value, torch.Tensor)
            )
            pinned_bytes = sum(
                value.numel() * value.element_size()
                for value in values
                if isinstance(value, torch.Tensor) and value.is_pinned()
            )
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            wall_started = time.perf_counter()
            start_event.record(torch.cuda.current_stream(device))
            moved = original_move(state_dict)
            end_event.record(torch.cuda.current_stream(device))
            end_event.synchronize()
            wall_ms = (time.perf_counter() - wall_started) * 1000.0
            gpu_ms = start_event.elapsed_time(end_event)
            with profile_lock:
                profile_state["move_calls"] += 1
                profile_state["move_wall_ms"] += wall_ms
                profile_state["move_gpu_ms"] += gpu_ms
                profile_state["move_bytes"] += payload_bytes
                profile_state["move_pinned_bytes"] += pinned_bytes
            return moved

        if args.profile_repeats:
            model.load_layer_to_cpu = profiled_load
            model.move_layer_to_device = profiled_move
            for _ in range(args.profile_repeats):
                before = dict(profile_state)
                current, attention_mask, wall_ms, gpu_ms = step(
                    current,
                    attention_mask,
                )
                profile_rows.append(
                    {
                        "wall_ms": wall_ms,
                        "gpu_ms": gpu_ms,
                        **{
                            key: profile_state[key] - before[key]
                            for key in profile_state
                        },
                    }
                )

    generated_ids = torch.cat(generated, dim=-1)
    report = {
        "schema_version": 1,
        "implementation": "AirLLM",
        "airllm_version": importlib.metadata.version("airllm"),
        "checkpoint": str(args.checkpoint),
        "layer_shards_root": str(args.layer_shards_root),
        "split_before": split_before,
        "split_after": split_after,
        "split_created_by_run": not split_before["exists"],
        "initialization_seconds": init_seconds,
        "prompt": args.prompt,
        "prompt_tokens": int(input_ids.numel()),
        "dtype": str(model.running_dtype),
        "prefetching": bool(model.prefetching),
        "compression": model.compression,
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
        "before_init": before_init,
        "after_init": after_init,
        "measured_decode_io_delta": {
            "read_bytes": (
                measured_end["read_bytes"] - measured_start["read_bytes"]
            ),
            "write_bytes": (
                measured_end["write_bytes"] - measured_start["write_bytes"]
            ),
        },
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "generated_token_ids": generated_ids.tolist(),
        "first_eight_generated_token_ids": generated_ids[:, :8].tolist(),
        "generated_text": model.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
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
