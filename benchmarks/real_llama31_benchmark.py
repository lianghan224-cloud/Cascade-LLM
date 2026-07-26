#!/usr/bin/env python3
"""Benchmark real Llama-3.1-8B CPU-resident streaming configurations."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    DoubleBufferRuntime,
    Llama31DecodeExecutor,
    ResidentDeviceArena,
    VocabStreamingRuntime,
    build_llama31_8b_plan,
    create_weight_store,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        required=True,
    )
    parser.add_argument(
        "--granularity",
        choices=("matrix", "matrix_group", "layer"),
        required=True,
    )
    parser.add_argument(
        "--vocab-mode",
        choices=("resident", "streamed"),
        default="resident",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--slots", type=int, choices=(1, 2), required=True)
    parser.add_argument("--warmup-decode", type=int, default=1)
    parser.add_argument("--decode-repeats", type=int, default=7)
    parser.add_argument("--profile-repeats", type=int, default=3)
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


def main():
    args = parse_args()
    if args.decode_repeats < 1 or args.profile_repeats < 1:
        raise SystemExit("repeat counts must be positive")
    if args.warmup_decode < 0:
        raise SystemExit("--warmup-decode cannot be negative")
    config = AutoConfig.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    plan = build_llama31_8b_plan(
        args.granularity,
        tie_word_embeddings=bool(config.tie_word_embeddings),
        stream_vocab=(args.vocab_mode == "streamed"),
    )

    before_load = process_snapshot()
    store = create_weight_store(plan, args.weight_store)
    load_started = time.perf_counter()
    store.load_checkpoint(args.checkpoint)
    load_seconds = time.perf_counter() - load_started
    after_load = process_snapshot()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident_started = time.perf_counter()
    resident = ResidentDeviceArena(plan, store, device)
    resident_load_seconds = time.perf_counter() - resident_started
    runtime = DoubleBufferRuntime(
        plan,
        store,
        resident,
        device,
        slot_count=args.slots,
        profile=False,
    )
    vocab_runtime = None
    if plan.vocab is not None:
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

    def step(current):
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        wall_started = time.perf_counter()
        total_start.record(torch.cuda.current_stream(device))
        state = executor.begin(current)
        state = runtime.run(executor, state)
        state = executor.finish(state)
        total_end.record(torch.cuda.current_stream(device))
        total_end.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        gpu_ms = total_start.elapsed_time(total_end)
        next_token = state.topk_indices[..., 0]
        generated.append(next_token.detach().cpu())
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
        if vocab_runtime is not None:
            vocab_runtime.profile = True
        profiles = []
        vocab_profiles = []
        profile_total_wall_ms = []
        profile_total_gpu_ms = []
        for _ in range(args.profile_repeats):
            current, wall_ms, gpu_ms = step(current)
            profiles.append(dict(runtime.last_profile))
            if vocab_runtime is not None:
                vocab_profiles.append(dict(vocab_runtime.last_profile))
            profile_total_wall_ms.append(wall_ms)
            profile_total_gpu_ms.append(gpu_ms)

    generated_ids = torch.cat(generated, dim=-1)
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "prompt_tokens": int(encoded.input_ids.numel()),
        "weight_store": args.weight_store,
        "granularity": args.granularity,
        "vocab_mode": args.vocab_mode,
        "top_k": args.top_k,
        "slots": args.slots,
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
        "runtime": runtime.stats.as_dict(),
        "cpu_pinned_bytes": store.pinned_cpu_bytes,
        "vocab_extra_pinned_cpu_bytes": (
            vocab_runtime.extra_pinned_cpu_bytes
            if vocab_runtime is not None
            else 0
        ),
        "cpu_arena_is_pinned": bool(store.arena.is_pinned()),
        "staging_slots_are_pinned": [
            bool(slot.is_pinned())
            for slot in (getattr(store, "staging_slots", None) or [])
        ],
        "before_load": before_load,
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
        "generated_text": tokenizer.decode(
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
    store.close()


if __name__ == "__main__":
    main()
