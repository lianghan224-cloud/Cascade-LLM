#!/usr/bin/env python3
"""Unified, versioned benchmark entry point for Cascade-LLM."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch
from transformers import AutoConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    BENCHMARK_REPORT_SCHEMA_VERSION,
    BackendSelection,
    BenchmarkCase,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSuiteReport,
    ExecutionPolicy,
    KVCacheManager,
    Llama31DecodeExecutor,
    MemoryPlanner,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MixedVocabStreamingRuntime,
    MultiDtypeWeightStore,
    PlacementMode,
    adapter_for_config,
    build_backend_phase_plan,
    build_static_transformer_placement,
    hardware_metadata,
    median_metrics,
    metric_variability,
)


BACKEND_FOR_FORMAT = {
    "bf16": "bf16_linear",
    "fp16": "fp16_linear",
    "int8_dequant_bf16_fallback": "int8_dequant_bf16_fallback",
    "int8_dequant_fp16_fallback": "int8_dequant_fp16_fallback",
    "int4_dequant_bf16_fallback": "int4_dequant_bf16_fallback",
    "int4_dequant_fp16_fallback": "int4_dequant_fp16_fallback",
}
SELECTABLE_BACKENDS = tuple(
    sorted(
        set(BACKEND_FOR_FORMAT.values()).union(
            {"fused_w8a16", "fused_w4a16", "fused_w8a8"}
        )
    )
)


def parse_byte_size(value):
    text = str(value).strip().lower()
    multipliers = {
        "gib": 1024 ** 3,
        "mib": 1024 ** 2,
        "gb": 1000 ** 3,
        "mb": 1000 ** 2,
        "b": 1,
    }
    for suffix in ("gib", "mib", "gb", "mb", "b"):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            break
    else:
        number = text
        suffix = "b"
    try:
        result = int(float(number) * multipliers[suffix])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "invalid byte size {!r}".format(value)
        ) from error
    if result < 0:
        raise argparse.ArgumentTypeError("byte size cannot be negative")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local Llama-family checkpoints. Synthetic results are "
            "regression evidence, not real-model performance claims."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Repeat for BF16, FP16, INT8 and INT4 checkpoint variants.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--preset",
        choices=("smoke", "core", "full", "m6a"),
        default="core",
    )
    parser.add_argument(
        "--backend",
        choices=("checkpoint",) + SELECTABLE_BACKENDS,
        default="checkpoint",
        help="Explicit backend request; mismatches fail instead of falling back.",
    )
    parser.add_argument(
        "--provider", choices=("none", "cutlass"), default="none"
    )
    parser.add_argument("--provider-library")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--prefill-tokens", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--kv-block-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--vocab-chunk-mib", type=float, default=4.0)
    parser.add_argument(
        "--embedding-placement",
        choices=("auto", "resident", "streamed"),
        default="auto",
    )
    parser.add_argument(
        "--lm-head-placement",
        choices=("auto", "resident", "streamed"),
        default="auto",
    )
    parser.add_argument(
        "--gpu-resident-weight-budget",
        type=parse_byte_size,
        default=0,
        help="Static complete-prefix Transformer residency budget, e.g. 4GiB.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional prior benchmark JSON used for automatic deltas.",
    )
    parser.add_argument(
        "--ignore-memlock-limit",
        action="store_true",
        help=(
            "Treat only RLIMIT_MEMLOCK preflight failures as warnings. "
            "Use this only after verifying that the CUDA pinned allocator works."
        ),
    )
    return parser.parse_args()


def preset_axes(preset):
    if preset == "m6a":
        return (
            ("matrix", "matrix_group", "layer"),
            ("pinned_staging",),
            ("streamed",),
        )
    if preset == "smoke":
        return (
            ("matrix_group",),
            ("pinned_staging",),
            ("streamed",),
        )
    if preset == "core":
        return (
            ("matrix_group",),
            ("full_pinned", "pinned_staging"),
            ("resident", "streamed"),
        )
    return (
        ("matrix", "matrix_group", "layer"),
        ("full_pinned", "pinned_staging"),
        ("resident", "streamed"),
    )


def compute_dtype(plan):
    dtype = next(
        spec.compute_dtype
        for spec in plan.weights.values()
        if spec.role == "attention_q"
    )
    return torch.bfloat16 if dtype == "bfloat16" else torch.float16


def sum_profile(profiles, key):
    return float(
        sum(
            profile.get(key, 0.0)
            for profile in profiles
            if profile
        )
    )


def max_profile(profiles, key):
    return int(
        max(
            (profile.get(key, 0) for profile in profiles if profile),
            default=0,
        )
    )


def percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    position = (len(values) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def execute_request(
    runtime,
    resident,
    vocab,
    manager,
    config,
    args,
):
    handle = manager.allocate(
        args.prefill_tokens + args.decode_tokens
    )
    cache = manager.bind(handle)
    executor = Llama31DecodeExecutor(
        config,
        resident,
        vocab_runtime=vocab,
        kv_cache=cache,
        top_k=min(10, config.vocab_size),
        return_full_logits=False,
        kv_dtype=compute_dtype(runtime.plan),
    )
    profiles = []
    vocab_profiles = []
    decode_latencies = []
    try:
        prompt = torch.tensor(
            [
                [
                    1 if index == 0 else 4 + (
                        index % max(1, config.vocab_size - 4)
                    )
                    for index in range(args.prefill_tokens)
                ]
            ],
            dtype=torch.long,
            device=runtime.device,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            state = runtime.run(executor, executor.begin(prompt))
            finish_started = time.perf_counter()
            state = executor.finish(state)
            torch.cuda.synchronize(runtime.device)
            resident_lm_wall = (
                time.perf_counter() - finish_started
            ) * 1000.0
        ttft_ms = (time.perf_counter() - started) * 1000.0
        profiles.append(dict(runtime.last_profile or {}))
        vocab_profiles.append(
            dict(vocab.last_profile or {})
            if vocab is not None
            else {"wall_ms": resident_lm_wall}
        )
        for index in range(args.decode_tokens):
            token = torch.tensor(
                [[4 + (index % max(1, config.vocab_size - 4))]],
                dtype=torch.long,
                device=runtime.device,
            )
            started = time.perf_counter()
            with torch.inference_mode():
                state = runtime.run(executor, executor.begin(token))
                finish_started = time.perf_counter()
                state = executor.finish(state)
                torch.cuda.synchronize(runtime.device)
                resident_lm_wall = (
                    time.perf_counter() - finish_started
                ) * 1000.0
            decode_latencies.append(
                (time.perf_counter() - started) * 1000.0
            )
            profiles.append(dict(runtime.last_profile or {}))
            vocab_profiles.append(
                dict(vocab.last_profile or {})
                if vocab is not None
                else {"wall_ms": resident_lm_wall}
            )
        del state
    finally:
        executor.close()
        cache.close()
    if manager.allocated_blocks:
        raise RuntimeError("benchmark request leaked KV blocks")
    return profiles, vocab_profiles, ttft_ms, decode_latencies


def make_metrics(
    profiles,
    vocab_profiles,
    ttft_ms,
    decode_latencies,
    device,
    placement,
):
    latency_window = min(100, len(decode_latencies))
    first_window_ms = (
        sum(decode_latencies[:latency_window]) / latency_window
        if latency_window
        else 0.0
    )
    last_window_ms = (
        sum(decode_latencies[-latency_window:]) / latency_window
        if latency_window
        else 0.0
    )
    forward_count = max(1, len(profiles))
    transformer_h2d_bytes = int(
        sum_profile(profiles, "weight_h2d_bytes")
    )
    lm_head_h2d_bytes = int(sum_profile(vocab_profiles, "h2d_bytes"))
    staging_ms = sum_profile(profiles, "staging_event_sum_ms")
    h2d_ms = sum_profile(profiles, "h2d_event_sum_ms")
    compute_ms = sum_profile(profiles, "compute_event_sum_ms")
    transformer_wall_ms = sum_profile(profiles, "wall_ms")
    source_wait_ms = sum_profile(profiles, "source_prepare_wait_ms")
    compute_wait_ms = sum_profile(profiles, "ready_wait_ms")
    serial_ms = staging_ms + h2d_ms + compute_ms
    return BenchmarkMetrics(
        pageable_to_pinned_ms=staging_ms,
        combined_weight_scale_h2d_ms=h2d_ms,
        weight_h2d_bytes=transformer_h2d_bytes,
        scale_h2d_bytes=int(
            sum_profile(profiles, "scale_h2d_bytes")
        ),
        dequant_ms=sum_profile(profiles, "dequant_event_sum_ms"),
        gemm_ms=sum_profile(profiles, "gemm_event_sum_ms"),
        attention_ms=sum_profile(profiles, "attention_event_sum_ms"),
        embedding_ms=sum_profile(
            vocab_profiles, "embedding_wall_ms"
        ),
        lm_head_ms=sum_profile(vocab_profiles, "wall_ms"),
        source_wait_ms=source_wait_ms,
        compute_wait_ms=compute_wait_ms,
        ttft_ms=float(ttft_ms),
        decode_ms_per_token=(
            sum(decode_latencies) / len(decode_latencies)
            if decode_latencies
            else 0.0
        ),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        source_queue_max_depth=max_profile(
            profiles, "source_queue_max_depth"
        ),
        ready_queue_max_depth=max_profile(
            profiles, "ready_queue_max_depth"
        ),
        transformer_weight_h2d_bytes_per_forward=(
            transformer_h2d_bytes / forward_count
        ),
        lm_head_h2d_bytes_per_forward=(
            lm_head_h2d_bytes / forward_count
        ),
        effective_pageable_to_pinned_gbps=(
            0.0
            if staging_ms <= 0.0
            else transformer_h2d_bytes / (staging_ms * 1.0e6)
        ),
        effective_h2d_gbps=(
            0.0
            if h2d_ms <= 0.0
            else (
                transformer_h2d_bytes
                + int(sum_profile(profiles, "scale_h2d_bytes"))
            )
            / (h2d_ms * 1.0e6)
        ),
        source_stall_ratio=(
            0.0
            if transformer_wall_ms <= 0.0
            else min(1.0, source_wait_ms / transformer_wall_ms)
        ),
        compute_stall_ratio=(
            0.0
            if transformer_wall_ms <= 0.0
            else min(1.0, compute_wait_ms / transformer_wall_ms)
        ),
        overlap_ratio=(
            0.0
            if serial_ms <= 0.0
            else max(0.0, min(1.0, 1.0 - transformer_wall_ms / serial_ms))
        ),
        resident_weight_bytes=placement.resident_weight_bytes,
        streamed_weight_bytes=placement.streamed_weight_bytes,
        resident_hit_ratio=placement.resident_hit_ratio,
        transformer_compute_ms=compute_ms,
        decode_p50_ms=percentile(decode_latencies, 0.50),
        decode_p95_ms=percentile(decode_latencies, 0.95),
        decode_p99_ms=percentile(decode_latencies, 0.99),
        decode_first_window_ms=first_window_ms,
        decode_last_window_ms=last_window_ms,
        decode_latency_drift_ratio=(
            0.0
            if first_window_ms <= 0.0
            else (last_window_ms - first_window_ms) / first_window_ms
        ),
        decode_latency_window_tokens=latency_window,
    )


def benchmark_case(case, config, adapter, args):
    policy = ExecutionPolicy.from_config(
        config,
        granularity=case.granularity,
        cpu_weight_mode=case.weight_store,
        embedding_mode=PlacementMode(case.embedding_placement),
        lm_head_mode=PlacementMode(case.lm_head_placement),
        slot_count=case.slot_count,
        prefetch_depth=case.prefetch_depth,
        vocab_chunk_bytes=int(args.vocab_chunk_mib * 1024 * 1024),
        linear_backend=case.backend_requested,
    )
    plan = adapter.build_execution_plan(config, policy)
    placement = build_static_transformer_placement(
        plan, case.gpu_resident_weight_budget_bytes
    )
    validation = adapter.validate_checkpoint(
        case.checkpoint, config, policy
    )
    if not validation.ok:
        return BenchmarkResult(
            case=case,
            status="error",
            skip_reason=validation.format_errors(),
            actual_backends=(),
            fallback_backends=(),
            samples=(),
            median=None,
        )
    backend_phase_plan = build_backend_phase_plan(
        plan,
        BackendSelection(
            prefill=case.backend_requested,
            decode=case.backend_requested,
        ),
        device=args.device,
        prefill_m_values=(args.prefill_tokens,),
        decode_m_values=(1,),
    )
    geometry = adapter.build_geometry(config)
    preflight = MemoryPlanner(
        plan,
        geometry,
        policy=policy,
        max_context=args.prefill_tokens + args.decode_tokens,
        max_prefill_tokens=args.prefill_tokens,
        kv_block_size=args.kv_block_size,
        embedding_staging_rows=args.prefill_tokens,
        return_full_logits=False,
        transformer_placement=placement,
    ).preflight(args.device, raise_on_error=False)
    if args.ignore_memlock_limit:
        memlock_errors = tuple(
            item
            for item in preflight.errors
            if "RLIMIT_MEMLOCK" in item
        )
        if memlock_errors:
            preflight = replace(
                preflight,
                errors=tuple(
                    item
                    for item in preflight.errors
                    if item not in memlock_errors
                ),
                warnings=preflight.warnings
                + tuple(
                    "explicitly ignored after CUDA pinned-allocation probe: "
                    + item
                    for item in memlock_errors
                ),
            )
    if not preflight.ok:
        return BenchmarkResult(
            case=case,
            status="skipped",
            skip_reason="; ".join(preflight.errors),
            actual_backends=(),
            fallback_backends=(),
            samples=(),
            median=None,
        )
    samples = []
    actual_backends = set()
    fallback_backends = set()
    try:
        with ExitStack() as resources:
            store = resources.enter_context(
                MultiDtypeWeightStore(
                    plan,
                    case.weight_store,
                    staging_slot_count=case.slot_count,
                )
            )
            store.load_checkpoint(case.checkpoint)
            resident = resources.enter_context(
                MixedResidentDeviceArena(
                    plan,
                    store,
                    device=args.device,
                    transformer_placement=placement,
                )
            )
            runtime = resources.enter_context(
                MixedDtypeRuntime(
                    plan,
                    store,
                    resident,
                    args.device,
                    slot_count=case.slot_count,
                    prefetch_depth=case.prefetch_depth,
                    profile=True,
                )
            )
            runtime.configure_backend_phase_plan(backend_phase_plan)
            vocab = None
            if plan.vocab.stream_embedding or plan.vocab.stream_lm_head:
                vocab = resources.enter_context(
                    MixedVocabStreamingRuntime(
                        plan,
                        store,
                        runtime,
                        embedding_staging_rows=args.prefill_tokens,
                        profile=True,
                    )
                )
            blocks = int(
                math.ceil(
                    (args.prefill_tokens + args.decode_tokens)
                    / float(args.kv_block_size)
                )
            )
            manager = resources.enter_context(
                KVCacheManager(
                    layer_count=config.num_hidden_layers,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=(
                        config.hidden_size // config.num_attention_heads
                    ),
                    total_blocks=blocks,
                    block_size=args.kv_block_size,
                    max_batch_size=1,
                    dtype=compute_dtype(plan),
                    device=args.device,
                )
            )
            for _ in range(args.warmup):
                execute_request(
                    runtime,
                    resident,
                    vocab,
                    manager,
                    config,
                    args,
                )
            for _ in range(args.repeats):
                torch.cuda.reset_peak_memory_stats(args.device)
                result = execute_request(
                    runtime,
                    resident,
                    vocab,
                    manager,
                    config,
                    args,
                )
                profiles, vocab_profiles, ttft_ms, decode_latencies = result
                samples.append(
                    make_metrics(
                        profiles,
                        vocab_profiles,
                        ttft_ms,
                        decode_latencies,
                        args.device,
                        placement,
                    )
                )
                for profile in profiles:
                    actual_backends.update(profile.get("backends", ()))
                    fallback_backends.update(
                        profile.get("fallback_backends", ())
                    )
        if actual_backends != {case.backend_requested}:
            raise RuntimeError(
                "requested backend {}, actual backends {}".format(
                    case.backend_requested, sorted(actual_backends)
                )
            )
    except BaseException as error:
        return BenchmarkResult(
            case=case,
            status="error",
            skip_reason="{}: {}".format(type(error).__name__, error),
            actual_backends=tuple(sorted(actual_backends)),
            fallback_backends=tuple(sorted(fallback_backends)),
            samples=tuple(samples),
            median=median_metrics(samples),
        )
    variability = metric_variability(samples)
    latency_cv = max(
        variability["ttft_ms"]["coefficient_of_variation"],
        variability["decode_ms_per_token"]["coefficient_of_variation"],
    )
    return BenchmarkResult(
        case=case,
        status="ok",
        skip_reason=None,
        actual_backends=tuple(sorted(actual_backends)),
        fallback_backends=tuple(sorted(fallback_backends)),
        samples=tuple(samples),
        median=median_metrics(samples),
        variability={
            "metrics": variability,
            "latency_cv_threshold": 0.05,
            "max_latency_cv": latency_cv,
            "stable": latency_cv < 0.05,
        },
    )


def _baseline_key(case):
    return (
        case.weight_format,
        case.backend_requested,
        case.granularity,
        case.weight_store,
        case.embedding_placement,
        case.lm_head_placement,
        case.slot_count,
        case.prefetch_depth,
        case.prefill_tokens,
        case.decode_tokens,
    )


def compare_to_baseline(results, baseline_path):
    if baseline_path is None:
        return ()
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = BenchmarkSuiteReport.from_dict(payload)
    by_key = {
        _baseline_key(item.case): item
        for item in baseline.results
        if item.status == "ok" and item.median is not None
    }
    comparisons = []
    for item in results:
        previous = by_key.get(_baseline_key(item.case))
        if previous is None or item.status != "ok" or item.median is None:
            continue
        metrics = {}
        for name in (
            "ttft_ms",
            "decode_ms_per_token",
            "transformer_weight_h2d_bytes_per_forward",
            "lm_head_h2d_bytes_per_forward",
            "gpu_peak_allocated_bytes",
        ):
            old = float(getattr(previous.median, name))
            new = float(getattr(item.median, name))
            metrics[name] = {
                "baseline": old,
                "current": new,
                "change_ratio": None if old == 0.0 else new / old - 1.0,
            }
        comparisons.append(
            {
                "case_id": item.case.case_id,
                "baseline_case_id": previous.case.case_id,
                "metrics": metrics,
            }
        )
    return tuple(comparisons)


def main():
    args = parse_args()
    for name in (
        "slots",
        "prefetch_depth",
        "prefill_tokens",
        "decode_tokens",
        "repeats",
    ):
        if getattr(args, name) < 1:
            raise SystemExit("--{} must be positive".format(name.replace("_", "-")))
    if args.warmup < 0 or args.vocab_chunk_mib <= 0:
        raise SystemExit("warmup must be non-negative and chunk size positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the benchmark")
    if args.provider == "cutlass":
        from layer_streaming.providers.cutlass import (
            load_cutlass_w8a16_provider,
        )

        load_cutlass_w8a16_provider(args.provider_library)
    granularities, stores, vocab_modes = preset_axes(args.preset)
    results = []
    synthetic_only = True
    started = time.time()
    for checkpoint in args.checkpoint:
        config = AutoConfig.from_pretrained(
            checkpoint, local_files_only=True
        )
        if args.prefill_tokens + args.decode_tokens > config.max_position_embeddings:
            raise SystemExit(
                "{} benchmark length exceeds model max_position_embeddings".format(
                    checkpoint
                )
            )
        adapter = adapter_for_config(config)
        inferred = ExecutionPolicy.from_config(config)
        requested = (
            BACKEND_FOR_FORMAT[inferred.weight_format.value]
            if args.backend == "checkpoint"
            else args.backend
        )
        synthetic_only = synthetic_only and (
            (checkpoint / "generation_manifest.json").is_file()
        )
        for granularity, store, vocab_mode in itertools.product(
            granularities, stores, vocab_modes
        ):
            embedding_placement = (
                vocab_mode
                if args.embedding_placement == "auto"
                else args.embedding_placement
            )
            lm_head_placement = (
                vocab_mode
                if args.lm_head_placement == "auto"
                else args.lm_head_placement
            )
            effective_vocab_mode = (
                embedding_placement
                if embedding_placement == lm_head_placement
                else "mixed"
            )
            case = BenchmarkCase(
                checkpoint=str(checkpoint.resolve()),
                weight_format=inferred.weight_format.value,
                backend_requested=requested,
                granularity=granularity,
                weight_store=store,
                vocab_mode=effective_vocab_mode,
                slot_count=args.slots,
                prefetch_depth=args.prefetch_depth,
                prefill_tokens=args.prefill_tokens,
                decode_tokens=args.decode_tokens,
                embedding_placement=embedding_placement,
                lm_head_placement=lm_head_placement,
                gpu_resident_weight_budget_bytes=(
                    args.gpu_resident_weight_budget
                ),
            )
            try:
                result = benchmark_case(case, config, adapter, args)
            except BaseException as error:
                result = BenchmarkResult(
                    case=case,
                    status="error",
                    skip_reason="{}: {}".format(
                        type(error).__name__, error
                    ),
                    actual_backends=(),
                    fallback_backends=(),
                    samples=(),
                    median=None,
                )
            results.append(result)
            print(
                "{}: {}".format(
                    case.case_id,
                    result.status
                    if result.skip_reason is None
                    else "{} ({})".format(
                        result.status, result.skip_reason
                    ),
                )
            )
    report = BenchmarkSuiteReport(
        schema_version=BENCHMARK_REPORT_SCHEMA_VERSION,
        synthetic_only=synthetic_only,
        hardware=hardware_metadata(args.device),
        software={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "duration_seconds": time.time() - started,
            "preset": args.preset,
            "measurement_notes": [
                "weight and scale share one transfer-unit copy; time is combined",
                "weight_h2d_bytes and scale_h2d_bytes remain separately reported",
                "synthetic checkpoints are regression data, not real-model performance",
                "backend mismatch is an error; no silent fallback is allowed",
            ],
        },
        results=tuple(results),
        baseline_comparisons=compare_to_baseline(results, args.baseline),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.to_json() + "\n", encoding="utf-8")
    errors = [item for item in results if item.status == "error"]
    print(
        "wrote {} cases ({} error, {} skipped) to {}".format(
            len(results),
            len(errors),
            sum(item.status == "skipped" for item in results),
            args.output,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
