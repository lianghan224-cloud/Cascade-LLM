#!/usr/bin/env python3
"""Run repeatable long-duration tiny-checkpoint stability tests."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import threading
import time

import torch
from transformers import AutoConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    ExecutionPolicy,
    KVCacheManager,
    Llama31DecodeExecutor,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MixedVocabStreamingRuntime,
    MultiDtypeWeightStore,
    PlacementMode,
    StabilityThresholds,
    adapter_for_config,
    analyze_stability,
    capture_resource_snapshot,
)


SOAK_SUITE_SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stress the persistent runtime using a local tiny or real "
            "Llama-family checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging", "both"),
        default="both",
    )
    parser.add_argument(
        "--granularity",
        choices=("matrix", "matrix_group", "layer"),
        default="matrix_group",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=("resident", "streamed"),
        default="streamed",
    )
    parser.add_argument(
        "--lm-head-mode",
        choices=("resident", "streamed"),
        default="streamed",
    )
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=1000,
        help="Total measured tokens per weight-store mode.",
    )
    parser.add_argument("--load-cycles", type=int, default=1)
    parser.add_argument("--cache-cycles", type=int, default=10)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--kv-block-size", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--fault-recovery-cycles",
        type=int,
        default=1,
        help="Create, fail, close and replace this many runtimes per load cycle.",
    )
    parser.add_argument(
        "--cuda-allocated-drift-mib", type=float, default=8.0
    )
    parser.add_argument(
        "--cuda-reserved-drift-mib", type=float, default=64.0
    )
    parser.add_argument("--latency-tail-ratio", type=float, default=1.25)
    return parser.parse_args()


def distribute(total, count):
    if count < 1 or total < count:
        raise ValueError("total must be at least count and count must be positive")
    quotient, remainder = divmod(int(total), int(count))
    return [
        quotient + (1 if index < remainder else 0)
        for index in range(count)
    ]


def compute_dtype_for_plan(plan):
    name = next(
        spec.name
        for spec in plan.weights.values()
        if spec.role == "attention_q"
    )
    return (
        torch.bfloat16
        if plan.weights[name].compute_dtype == "bfloat16"
        else torch.float16
    )


def run_tokens(
    runtime,
    resident,
    vocab,
    manager,
    config,
    token_count,
    token_base,
    *,
    sample_every,
    sample_start,
    warmup=False,
):
    handle = manager.allocate(token_count)
    cache = manager.bind(handle)
    snapshots = []
    latencies = []
    executor = Llama31DecodeExecutor(
        config,
        resident,
        vocab_runtime=vocab,
        kv_cache=cache,
        top_k=min(10, config.vocab_size),
        return_full_logits=False,
        kv_dtype=compute_dtype_for_plan(runtime.plan),
    )
    try:
        with torch.inference_mode():
            for local_index in range(token_count):
                token_index = token_base + local_index
                token_id = 1 if local_index == 0 else 4 + (
                    token_index % max(1, config.vocab_size - 4)
                )
                ids = torch.tensor(
                    [[token_id]], dtype=torch.long, device=runtime.device
                )
                started = time.perf_counter()
                state = executor.finish(
                    runtime.run(executor, executor.begin(ids))
                )
                del state
                torch.cuda.synchronize(runtime.device)
                elapsed = (time.perf_counter() - started) * 1000.0
                if warmup:
                    continue
                latencies.append(elapsed)
                if (
                    (token_index + 1) % sample_every == 0
                    or local_index + 1 == token_count
                ):
                    snapshots.append(
                        capture_resource_snapshot(
                            sample_start + len(snapshots),
                            token_index + 1,
                            runtime=runtime,
                            store=runtime.store,
                            kv_manager=manager,
                            vocab_runtime=vocab,
                            token_latency_ms=elapsed,
                        )
                    )
    finally:
        executor.close()
        cache.close()
    if manager.allocated_blocks or manager.resource_stats()["active_handles"]:
        raise RuntimeError("KV blocks remained allocated after request release")
    return snapshots, latencies


def build_policy(config, args, mode):
    inferred = ExecutionPolicy.from_config(config)
    return ExecutionPolicy.from_config(
        config,
        granularity=args.granularity,
        cpu_weight_mode=mode,
        embedding_mode=PlacementMode(args.embedding_mode),
        lm_head_mode=PlacementMode(args.lm_head_mode),
        slot_count=args.slots,
        prefetch_depth=args.prefetch_depth,
        vocab_chunk_bytes=4 * 1024 * 1024,
        weight_format=inferred.weight_format,
        quantization=inferred.quantization,
    )


def inject_compute_failures(plan, store, resident, args):
    for _ in range(args.fault_recovery_cycles):
        before = {
            thread.ident
            for thread in threading.enumerate()
            if thread.is_alive()
        }
        runtime = MixedDtypeRuntime(
            plan,
            store,
            resident,
            args.device,
            slot_count=args.slots,
            prefetch_depth=args.prefetch_depth,
        )

        def fail_on_compute(unit, weights, state):
            del unit, weights, state
            raise RuntimeError("intentional soak recovery failure")

        try:
            runtime.run(fail_on_compute, None)
        except RuntimeError as error:
            if "intentional soak recovery failure" not in str(error):
                raise
        else:
            raise RuntimeError("intentional compute failure was not propagated")
        finally:
            runtime.close()
            runtime.close()
        leaked = [
            thread.name
            for thread in threading.enumerate()
            if thread.is_alive()
            and thread.ident not in before
            and thread.name.startswith("cascade-")
        ]
        if leaked:
            raise RuntimeError(
                "pipeline workers survived failed runtime close: {}".format(
                    leaked
                )
            )


def run_mode(args, config, adapter, mode):
    policy = build_policy(config, args, mode)
    plan = adapter.build_execution_plan(config, policy)
    adapter.validate_checkpoint(
        args.checkpoint, config, policy
    ).raise_for_error()
    token_cycles = distribute(args.decode_tokens, args.cache_cycles)
    cache_cycles_per_load = distribute(args.cache_cycles, args.load_cycles)
    snapshots = []
    token_latencies = []
    token_offset = 0
    cache_offset = 0
    max_queue_depths = {"source": 0, "ready": 0}
    baseline_cascade_threads = sum(
        thread.is_alive()
        and (
            thread.name.startswith("cascade-")
            or thread.name.startswith("mixed-weight-stage")
        )
        for thread in threading.enumerate()
    )
    for load_index, cycle_count in enumerate(cache_cycles_per_load):
        with ExitStack() as resources:
            store = resources.enter_context(
                MultiDtypeWeightStore(
                    plan,
                    mode,
                    staging_slot_count=args.slots,
                )
            )
            store.load_checkpoint(args.checkpoint)
            resident = resources.enter_context(
                MixedResidentDeviceArena(
                    plan, store, device=args.device
                )
            )
            inject_compute_failures(plan, store, resident, args)
            runtime = resources.enter_context(
                MixedDtypeRuntime(
                    plan,
                    store,
                    resident,
                    args.device,
                    slot_count=args.slots,
                    prefetch_depth=args.prefetch_depth,
                    profile=False,
                )
            )
            vocab = None
            if plan.vocab.stream_embedding or plan.vocab.stream_lm_head:
                vocab = resources.enter_context(
                    MixedVocabStreamingRuntime(
                        plan,
                        store,
                        runtime,
                        embedding_staging_rows=1,
                    )
                )
            maximum_request = max(
                token_cycles[
                    cache_offset : cache_offset + cycle_count
                ]
                + [args.warmup_tokens]
            )
            manager = resources.enter_context(
                KVCacheManager(
                    layer_count=config.num_hidden_layers,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=(
                        config.hidden_size // config.num_attention_heads
                    ),
                    total_blocks=int(
                        math.ceil(
                            maximum_request / float(args.kv_block_size)
                        )
                    ),
                    block_size=args.kv_block_size,
                    max_batch_size=1,
                    dtype=compute_dtype_for_plan(plan),
                    device=args.device,
                )
            )
            if args.warmup_tokens:
                run_tokens(
                    runtime,
                    resident,
                    vocab,
                    manager,
                    config,
                    args.warmup_tokens,
                    0,
                    sample_every=max(1, args.warmup_tokens),
                    sample_start=0,
                    warmup=True,
                )
            for request_tokens in token_cycles[
                cache_offset : cache_offset + cycle_count
            ]:
                new_snapshots, new_latencies = run_tokens(
                    runtime,
                    resident,
                    vocab,
                    manager,
                    config,
                    request_tokens,
                    token_offset,
                    sample_every=args.sample_every,
                    sample_start=len(snapshots),
                )
                snapshots.extend(new_snapshots)
                token_latencies.extend(new_latencies)
                token_offset += request_tokens
                profile = runtime.last_profile or {}
                max_queue_depths["source"] = max(
                    max_queue_depths["source"],
                    int(profile.get("source_queue_max_depth", 0)),
                )
                max_queue_depths["ready"] = max(
                    max_queue_depths["ready"],
                    int(profile.get("ready_queue_max_depth", 0)),
                )
            cache_offset += cycle_count
        torch.cuda.synchronize(args.device)
        torch.cuda.empty_cache()
    final_cascade_threads = sum(
        thread.is_alive()
        and (
            thread.name.startswith("cascade-")
            or thread.name.startswith("mixed-weight-stage")
        )
        for thread in threading.enumerate()
    )
    if final_cascade_threads != baseline_cascade_threads:
        raise RuntimeError(
            "cascade thread count changed from {} to {} after lifecycle test".format(
                baseline_cascade_threads, final_cascade_threads
            )
        )
    if len(snapshots) < 2:
        raise RuntimeError(
            "soak produced fewer than two samples; lower --sample-every"
        )
    thresholds = StabilityThresholds(
        cuda_allocated_drift_bytes=int(
            args.cuda_allocated_drift_mib * 1024 * 1024
        ),
        cuda_reserved_drift_bytes=int(
            args.cuda_reserved_drift_mib * 1024 * 1024
        ),
        latency_tail_ratio=args.latency_tail_ratio,
    )
    report = analyze_stability(snapshots, thresholds=thresholds)
    return {
        "mode": mode,
        "weight_format": policy.weight_format.value,
        "granularity": policy.granularity.value,
        "embedding_mode": policy.embedding_mode.value,
        "lm_head_mode": policy.lm_head_mode.value,
        "decode_tokens": len(token_latencies),
        "load_cycles": args.load_cycles,
        "cache_cycles": args.cache_cycles,
        "fault_recovery_cycles_per_load": args.fault_recovery_cycles,
        "max_queue_depths": max_queue_depths,
        "stability": report.as_dict(),
    }


def main():
    args = parse_args()
    for name in (
        "decode_tokens",
        "load_cycles",
        "cache_cycles",
        "sample_every",
        "slots",
        "prefetch_depth",
    ):
        if getattr(args, name) < 1:
            raise SystemExit("--{} must be positive".format(name.replace("_", "-")))
    if args.decode_tokens < args.cache_cycles:
        raise SystemExit("--decode-tokens must be at least --cache-cycles")
    if args.cache_cycles < args.load_cycles:
        raise SystemExit("--cache-cycles must be at least --load-cycles")
    if args.warmup_tokens < 0 or args.fault_recovery_cycles < 0:
        raise SystemExit("warmup and fault-recovery counts cannot be negative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the soak tool")
    config = AutoConfig.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    adapter = adapter_for_config(config)
    modes = (
        ("full_pinned", "pinned_staging")
        if args.weight_store == "both"
        else (args.weight_store,)
    )
    started = time.time()
    results = [
        run_mode(args, config, adapter, mode)
        for mode in modes
    ]
    output = {
        "schema_version": SOAK_SUITE_SCHEMA_VERSION,
        "synthetic_only": (
            getattr(config, "_name_or_path", "")
            == "cascade-tiny-synthetic"
            or (args.checkpoint / "generation_manifest.json").is_file()
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device,
        "started_unix_seconds": started,
        "duration_seconds": time.time() - started,
        "passed": all(item["stability"]["passed"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": output["passed"],
                "synthetic_only": output["synthetic_only"],
                "duration_seconds": output["duration_seconds"],
                "output": str(args.output),
                "results": [
                    {
                        "mode": item["mode"],
                        "passed": item["stability"]["passed"],
                        "samples": item["stability"]["sample_count"],
                        "failures": item["stability"]["failures"],
                        "latency": item["stability"]["latency"],
                    }
                    for item in results
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
