#!/usr/bin/env python3
"""Real single-GPU benchmark for Llama 3.1 70B Instruct W8A8."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import psutil
import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    Int8DoubleBufferRuntime,
    Int8ResidentDeviceArena,
    Llama31DecodeExecutor,
    VocabStreamingRuntime,
    build_llama31_70b_int8_plan,
    create_int8_weight_store,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="pinned_staging",
    )
    parser.add_argument(
        "--granularity",
        choices=("matrix", "matrix_group", "layer"),
        default="matrix",
    )
    parser.add_argument("--slots", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmup-decode", type=int, default=0)
    parser.add_argument("--decode-repeats", type=int, default=1)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--include-plan-units", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def process_snapshot():
    process = psutil.Process()
    memory = process.memory_info()
    io = process.io_counters()
    return {
        "rss_bytes": memory.rss,
        "vms_bytes": memory.vms,
        "read_bytes": io.read_bytes,
        "write_bytes": io.write_bytes,
        "system_available_bytes": psutil.virtual_memory().available,
    }


def distribution(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
    }


def nvidia_smi():
    import subprocess

    command = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,name,memory.total,memory.used,"
        "memory.free,pstate,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main():
    args = parse_args()
    if args.decode_repeats < 1 or args.profile_repeats < 1:
        raise SystemExit("decode and profile repeats must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    config = AutoConfig.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    quantization = getattr(config, "quantization_config", None)
    if not quantization:
        raise SystemExit("checkpoint has no quantization_config")
    if quantization.get("quant_method") != "compressed-tensors":
        raise SystemExit("checkpoint is not compressed-tensors")

    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    plan = build_llama31_70b_int8_plan(args.granularity)

    before_load = process_snapshot()
    allocation_started = time.perf_counter()
    store = create_int8_weight_store(
        plan,
        args.weight_store,
        slot_count=args.slots,
    )
    allocation_seconds = time.perf_counter() - allocation_started
    after_allocate = process_snapshot()
    load_started = time.perf_counter()
    store.load_checkpoint(args.checkpoint)
    load_seconds = time.perf_counter() - load_started
    after_load = process_snapshot()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident_started = time.perf_counter()
    resident = Int8ResidentDeviceArena(plan, store, device)
    resident_load_seconds = time.perf_counter() - resident_started
    runtime = Int8DoubleBufferRuntime(
        plan,
        store,
        resident,
        device=device,
        slot_count=args.slots,
        profile=False,
    )
    vocab_runtime = VocabStreamingRuntime(
        plan,
        store,
        runtime,
        profile=False,
    )
    executor = Llama31DecodeExecutor(
        config,
        resident,
        vocab_runtime=vocab_runtime,
        top_k=args.top_k,
    )
    after_runtime_init = process_snapshot()

    generated = []
    generated_topk = []

    def step(current):
        torch.cuda.synchronize(device)
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        gpu_start.record(torch.cuda.current_stream(device))
        state = executor.begin(current)
        state = runtime.run(executor, state)
        state = executor.finish(state)
        gpu_end.record(torch.cuda.current_stream(device))
        gpu_end.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        gpu_ms = gpu_start.elapsed_time(gpu_end)
        next_token = state.topk_indices[..., 0]
        generated.append(next_token.detach().cpu())
        generated_topk.append(
            {
                "values": state.topk_values.float().detach().cpu().tolist(),
                "indices": state.topk_indices.detach().cpu().tolist(),
            }
        )
        return next_token, wall_ms, gpu_ms

    with torch.inference_mode():
        current = encoded.input_ids.to(device)
        current, prefill_wall_ms, prefill_gpu_ms = step(current)
        for _ in range(args.warmup_decode):
            current, _, _ = step(current)

        measured_io_start = process_snapshot()
        decode_wall_ms = []
        decode_gpu_ms = []
        for _ in range(args.decode_repeats):
            current, wall_ms, gpu_ms = step(current)
            decode_wall_ms.append(wall_ms)
            decode_gpu_ms.append(gpu_ms)
        measured_io_end = process_snapshot()

        runtime.profile = True
        vocab_runtime.profile = True
        profiles = []
        vocab_profiles = []
        profile_total_wall_ms = []
        profile_total_gpu_ms = []
        for _ in range(args.profile_repeats):
            current, wall_ms, gpu_ms = step(current)
            profiles.append(dict(runtime.last_profile))
            vocab_profiles.append(dict(vocab_runtime.last_profile))
            profile_total_wall_ms.append(wall_ms)
            profile_total_gpu_ms.append(gpu_ms)

    generated_ids = torch.cat(generated, dim=-1)
    runtime_stats = dict(runtime.stats)
    transformer_h2d = plan.stream_bytes_per_token
    lm_head_h2d = plan.vocab.vocab_size * plan.vocab.hidden_size * 2
    report = {
        "schema_version": 1,
        "benchmark": "real_llama31_70b_int8_streaming",
        "checkpoint": str(args.checkpoint),
        "model_id": plan.model_id,
        "prompt": args.prompt,
        "prompt_tokens": int(encoded.input_ids.numel()),
        "weight_store": args.weight_store,
        "granularity": args.granularity,
        "slots": args.slots,
        "top_k": args.top_k,
        "allocation_seconds": allocation_seconds,
        "load_seconds": load_seconds,
        "resident_load_seconds": resident_load_seconds,
        "prefill_wall_ms": prefill_wall_ms,
        "prefill_gpu_ms": prefill_gpu_ms,
        "warmup_decode": args.warmup_decode,
        "decode_repeats": args.decode_repeats,
        "decode_wall_ms": decode_wall_ms,
        "decode_gpu_ms": decode_gpu_ms,
        "decode_wall_distribution_ms": distribution(decode_wall_ms),
        "decode_gpu_distribution_ms": distribution(decode_gpu_ms),
        "profile_repeats": args.profile_repeats,
        "profile_total_wall_ms": profile_total_wall_ms,
        "profile_total_gpu_ms": profile_total_gpu_ms,
        "profiles": profiles,
        "vocab_profiles": vocab_profiles,
        "runtime": runtime_stats,
        "plan": plan.as_dict(
            include_units=args.include_plan_units
        ),
        "h2d_accounting": {
            "transformer_bytes_per_token": transformer_h2d,
            "lm_head_bytes_per_token": lm_head_h2d,
            "embedding_bytes_last_step": vocab_runtime.last_embedding_bytes,
            "total_decode_weight_bytes": transformer_h2d + lm_head_h2d,
        },
        "cpu_pinned_bytes": store.pinned_cpu_bytes,
        "vocab_extra_pinned_cpu_bytes": (
            vocab_runtime.extra_pinned_cpu_bytes
        ),
        "cpu_arena_is_pinned": bool(store.arena_is_pinned),
        "staging_slots_are_pinned": [
            bool(slot.is_pinned())
            for slot in (getattr(store, "staging_slots", None) or [])
        ],
        "before_load": before_load,
        "after_allocate": after_allocate,
        "after_load": after_load,
        "after_runtime_init": after_runtime_init,
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
        "generated_topk": generated_topk,
        "generated_text": tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ),
        "quantization": {
            "quant_method": quantization.get("quant_method"),
            "format": quantization.get("format"),
            "ignore": quantization.get("ignore"),
            "execution_note": (
                "INT8 weights are transferred as stored and dequantized "
                "per matrix to a reusable BF16 GPU workspace; activation "
                "quantization is not used by this runtime."
            ),
        },
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": nvidia_smi(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    store.close()


if __name__ == "__main__":
    main()
