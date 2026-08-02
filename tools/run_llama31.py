#!/usr/bin/env python3
"""Run a local Llama-family checkpoint through the generic streaming runtime."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    BackendSelection,
    CompatibilityRequest,
    CompatibilityResolver,
    ExecutionPolicy,
    HardwareDetector,
    KVPolicy,
    Llama31DecodeExecutor,
    MemoryPlanner,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MixedVocabStreamingRuntime,
    MultiDtypeWeightStore,
    PlacementMode,
    QuantizationSpec,
    WeightFormat,
    adapter_for_config,
    build_backend_phase_plan,
    build_compatibility_report,
    build_inference_report,
    build_static_transformer_placement,
    default_provider_registry,
)


BACKEND_CHOICES = (
    "checkpoint",
    "bf16_linear",
    "fp16_linear",
    "int8_dequant_bf16_fallback",
    "int8_dequant_fp16_fallback",
    "int4_dequant_bf16_fallback",
    "int4_dequant_fp16_fallback",
    "fused_w8a16",
    "fused_w4a16",
    "fused_w8a8",
)


def parse_byte_size(value):
    text = str(value).strip().lower()
    multiplier = 1
    for suffix, amount in (("gib", 1024 ** 3), ("mib", 1024 ** 2), ("b", 1)):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            multiplier = amount
            break
    try:
        result = int(float(text) * multiplier)
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid byte size") from error
    if result < 0:
        raise argparse.ArgumentTypeError("byte size cannot be negative")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--weight-format",
        choices=("auto",) + tuple(item.value for item in WeightFormat),
        default="auto",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="checkpoint",
        help="Explicit LinearBackend; unavailable fused providers fail early.",
    )
    parser.add_argument("--prefill-backend", choices=BACKEND_CHOICES)
    parser.add_argument("--decode-backend", choices=BACKEND_CHOICES)
    parser.add_argument(
        "--provider", choices=("none", "cutlass"), default="none"
    )
    parser.add_argument("--provider-library")
    parser.add_argument(
        "--embedding-dtype", choices=("auto", "bf16", "fp16"), default="auto"
    )
    parser.add_argument(
        "--lm-head-dtype", choices=("auto", "bf16", "fp16"), default="auto"
    )
    parser.add_argument(
        "--norm-dtype", choices=("auto", "bf16", "fp16"), default="auto"
    )
    parser.add_argument(
        "--quant-granularity",
        choices=("auto", "per_channel", "per_group"),
        default="auto",
    )
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128))
    parser.add_argument(
        "--scale-dtype", choices=("auto", "bf16", "fp16"), default="auto"
    )
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="full_pinned",
    )
    parser.add_argument(
        "--granularity",
        choices=("matrix", "matrix_group", "layer"),
        default="matrix_group",
    )
    parser.add_argument(
        "--vocab-mode",
        choices=("resident", "streamed"),
        help="Deprecated shorthand that sets both vocabulary placements",
    )
    parser.add_argument(
        "--embedding-mode",
        "--embedding-placement",
        dest="embedding_mode",
        choices=("auto", "resident", "streamed"),
        default="auto",
    )
    parser.add_argument(
        "--lm-head-mode",
        "--lm-head-placement",
        dest="lm_head_mode",
        choices=("auto", "resident", "streamed"),
        default="auto",
    )
    parser.add_argument("--slots", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--prefetch-depth", type=int, choices=tuple(range(1, 9))
    )
    parser.add_argument(
        "--kv-block-size",
        "--kv-page-size",
        dest="kv_block_size",
        type=int,
        choices=(16, 32),
        default=16,
    )
    parser.add_argument(
        "--kv-accuracy",
        choices=("exact", "quantized", "sparse"),
        default="exact",
    )
    parser.add_argument(
        "--kv-storage",
        choices=("gpu", "gpu-cpu", "gpu-cpu-nvme"),
        default="gpu",
    )
    parser.add_argument(
        "--kv-dtype",
        choices=("auto", "bf16", "fp16", "int8", "fp8", "int4"),
        default="auto",
    )
    parser.add_argument(
        "--kv-index",
        choices=("none", "quest-flat", "hierarchical-quest", "centroid-only"),
        default="none",
    )
    parser.add_argument(
        "--kv-prefix-cache",
        choices=("off", "session", "memory", "persistent"),
        default="off",
    )
    parser.add_argument("--kv-cpu-budget", type=parse_byte_size, default=0)
    parser.add_argument("--kv-nvme-budget", type=parse_byte_size, default=0)
    parser.add_argument("--kv-page-budget", type=int, default=0)
    parser.add_argument("--kv-recent-window", type=int, default=0)
    parser.add_argument("--return-full-logits", action="store_true")
    parser.add_argument(
        "--no-profile",
        action="store_false",
        dest="profile",
        default=True,
        help="Disable CUDA stage timing in the JSON report",
    )
    parser.add_argument("--cuda-safety-margin-mib", type=int, default=512)
    parser.add_argument(
        "--ignore-memlock-limit",
        action="store_true",
        help=(
            "Treat only RLIMIT_MEMLOCK preflight failures as warnings. "
            "Use only after confirming CUDA pinned allocation works."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gpu-resident-weight-budget",
        type=parse_byte_size,
        default=0,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _compatibility_weight_format(weight):
    quant = weight.quantization
    if quant is None:
        return "bf16" if weight.storage_dtype == "bfloat16" else "fp16"
    return "int{}_{}_{}".format(
        quant.bits, quant.scheme, quant.granularity
    )


def _build_compatibility_requests(plan, selection, prompt_m):
    requests = []
    for phase, backend, m in (
        ("prefill", selection.prefill, int(prompt_m)),
        ("decode", selection.decode, 1),
    ):
        for weight in plan.weights.values():
            if weight.role not in {
                "attention_q",
                "attention_k",
                "attention_v",
                "attention_o",
                "mlp_gate",
                "mlp_up",
                "mlp_down",
            }:
                continue
            quant = weight.quantization
            n, k = (int(item) for item in weight.logical_shape)
            requests.append(
                (
                    phase,
                    weight.name,
                    CompatibilityRequest(
                        phase=phase,
                        backend_requested=backend,
                        weight_format=_compatibility_weight_format(weight),
                        activation_dtype=(
                            "bf16"
                            if weight.compute_dtype == "bfloat16"
                            else "fp16"
                        ),
                        scale_dtype=(
                            None
                            if quant is None
                            else (
                                "bf16"
                                if quant.scale_dtype == "bfloat16"
                                else "fp16"
                            )
                        ),
                        group_size=None if quant is None else quant.group_size,
                        m=m,
                        n=n,
                        k=k,
                        physical_layout="row_major",
                        workspace_limit_bytes=0,
                    ),
                )
            )
    return tuple(requests)


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.provider == "cutlass":
        from layer_streaming.providers.cutlass import (
            load_cutlass_w8a16_provider,
        )

        load_cutlass_w8a16_provider(args.provider_library)

    config = AutoConfig.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    embedding_mode = (
        "streamed" if args.embedding_mode == "auto" else args.embedding_mode
    )
    lm_head_mode = (
        "streamed" if args.lm_head_mode == "auto" else args.lm_head_mode
    )
    if args.vocab_mode is not None:
        embedding_mode = args.vocab_mode
        lm_head_mode = args.vocab_mode
    prefetch_depth = args.prefetch_depth or args.slots
    inferred = ExecutionPolicy.from_config(config)
    weight_format = (
        inferred.weight_format
        if args.weight_format == "auto"
        else WeightFormat(args.weight_format)
    )
    quantization = inferred.quantization
    if "int8" in weight_format.value or "int4" in weight_format.value:
        bits = 8 if "int8" in weight_format.value else 4
        if quantization is None or quantization.bits != bits:
            quantization = QuantizationSpec(
                bits=bits,
                granularity="per_channel" if bits == 8 else "per_group",
                group_size=None if bits == 8 else (args.group_size or 64),
            )
        granularity = (
            quantization.granularity
            if args.quant_granularity == "auto"
            else args.quant_granularity
        )
        group_size = (
            args.group_size
            if granularity == "per_group"
            else None
        )
        if granularity == "per_group" and group_size is None:
            group_size = quantization.group_size or 64
        quantization = QuantizationSpec(
            bits=bits,
            scheme=quantization.scheme,
            granularity=granularity,
            group_size=group_size,
            scale_dtype=(
                quantization.scale_dtype
                if args.scale_dtype == "auto"
                else args.scale_dtype
            ),
            zero_point=quantization.zero_point,
            zero_point_dtype=quantization.zero_point_dtype,
            packing=quantization.packing,
            axis=quantization.axis,
        )
    else:
        quantization = None
    checkpoint_backend = {
        WeightFormat.BF16: "bf16_linear",
        WeightFormat.FP16: "fp16_linear",
    }.get(weight_format, weight_format.value)
    phase_selection = None
    if args.prefill_backend is not None or args.decode_backend is not None:
        prefill_backend = args.prefill_backend or args.backend
        decode_backend = args.decode_backend or args.backend
        prefill_backend = (
            checkpoint_backend
            if prefill_backend == "checkpoint"
            else prefill_backend
        )
        decode_backend = (
            checkpoint_backend
            if decode_backend == "checkpoint"
            else decode_backend
        )
        phase_selection = BackendSelection(
            prefill=prefill_backend,
            decode=decode_backend,
        )
        plan_backend = next(
            (
                name
                for name in (prefill_backend, decode_backend)
                if "fallback" in name
            ),
            prefill_backend,
        )
    else:
        plan_backend = None if args.backend == "checkpoint" else args.backend
        if plan_backend is not None:
            phase_selection = BackendSelection(
                prefill=plan_backend,
                decode=plan_backend,
            )
    policy = ExecutionPolicy(
        granularity=args.granularity,
        weight_format=weight_format,
        cpu_weight_mode=args.weight_store,
        embedding_mode=PlacementMode(embedding_mode),
        lm_head_mode=PlacementMode(lm_head_mode),
        slot_count=args.slots,
        prefetch_depth=prefetch_depth,
        embedding_dtype=(
            inferred.embedding_dtype
            if args.embedding_dtype == "auto"
            else args.embedding_dtype
        ),
        lm_head_dtype=(
            inferred.lm_head_dtype
            if args.lm_head_dtype == "auto"
            else args.lm_head_dtype
        ),
        norm_dtype=(
            inferred.norm_dtype
            if args.norm_dtype == "auto"
            else args.norm_dtype
        ),
        quantization=quantization,
        linear_backend=plan_backend,
    )
    adapter = adapter_for_config(config)
    geometry = adapter.build_geometry(config)
    plan = adapter.build_execution_plan(config, policy)
    compute_dtype = next(
        spec.compute_dtype
        for spec in plan.weights.values()
        if spec.role == "attention_q"
    )
    kv_dtype_name = (
        ("bf16" if compute_dtype == "bfloat16" else "fp16")
        if args.kv_dtype == "auto"
        else args.kv_dtype
    )
    kv_reuse = {
        "off": "none",
        "session": "session",
        "memory": "prefix_memory",
        "persistent": "prefix_persistent",
    }[args.kv_prefix_cache]
    try:
        kv_policy = KVPolicy(
            accuracy=args.kv_accuracy,
            storage=args.kv_storage.replace("-", "_"),
            dtype=kv_dtype_name,
            selection=args.kv_index.replace("-", "_"),
            reuse=kv_reuse,
            page_size=args.kv_block_size,
            cpu_budget_bytes=args.kv_cpu_budget,
            nvme_budget_bytes=args.kv_nvme_budget,
            page_budget=args.kv_page_budget,
            recent_window=args.kv_recent_window,
        )
        kv_policy.require_d1_supported()
    except (ValueError, NotImplementedError) as error:
        raise SystemExit("KV policy rejected before allocation: {}".format(error))
    print(
        "Expanded KV policy: {}".format(
            json.dumps(kv_policy.as_dict(), sort_keys=True)
        ),
        file=sys.stderr,
    )
    transformer_placement = build_static_transformer_placement(
        plan, args.gpu_resident_weight_budget
    )
    planned_backends = {
        tensor.backend
        for unit in plan.units
        for tensor in unit.tensors
        if tensor.backend
    }
    if plan_backend is not None and planned_backends != {plan_backend}:
        raise SystemExit(
            "requested backend {}, execution plan uses {}".format(
                plan_backend, sorted(planned_backends)
            )
        )
    validation = adapter.validate_checkpoint(
        args.checkpoint, config, policy
    ).raise_for_error()
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    cache_length = int(encoded.input_ids.shape[1]) + args.max_new_tokens
    if cache_length > geometry.max_position_embeddings:
        raise SystemExit(
            "prompt plus output exceeds model context limit {}".format(
                geometry.max_position_embeddings
            )
        )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    compatibility_report = None
    if phase_selection is not None:
        hardware, runtime_features = HardwareDetector().detect(
            device, refresh=True
        )
        compatibility_registry = default_provider_registry()
        compatibility_resolver = CompatibilityResolver(
            compatibility_registry
        )
        compatibility_requests = _build_compatibility_requests(
            plan, phase_selection, int(encoded.input_ids.numel())
        )
        compatibility_decisions = tuple(
            compatibility_resolver.resolve(
                hardware, runtime_features, request
            )
            for _, _, request in compatibility_requests
        )
        compatibility_report = build_compatibility_report(
            hardware,
            runtime_features,
            compatibility_registry,
            decisions=compatibility_decisions,
            selected_backend="prefill={},decode={}".format(
                phase_selection.prefill, phase_selection.decode
            ),
        )
        unsupported = tuple(
            (phase, name, decision)
            for (phase, name, _), decision in zip(
                compatibility_requests, compatibility_decisions
            )
            if not decision.supported
        )
        if unsupported:
            phase, name, decision = unsupported[0]
            raise SystemExit(
                "hardware compatibility rejected {} {}: {}".format(
                    phase, name, "; ".join(decision.reasons)
                )
            )
    backend_phase_plan = (
        build_backend_phase_plan(
            plan,
            phase_selection,
            device=device,
            prefill_m_values=(int(encoded.input_ids.numel()),),
            decode_m_values=(1,),
        )
        if phase_selection is not None
        else None
    )
    preflight = MemoryPlanner(
        plan,
        geometry,
        policy=policy,
        max_context=cache_length,
        max_prefill_tokens=int(encoded.input_ids.shape[1]),
        kv_block_size=args.kv_block_size,
        embedding_staging_rows=max(1, int(encoded.input_ids.numel())),
        return_full_logits=args.return_full_logits,
        cuda_safety_margin_bytes=args.cuda_safety_margin_mib * 1024 ** 2,
        transformer_placement=transformer_placement,
        kv_policy=kv_policy,
    ).preflight(device=device, raise_on_error=False)
    if args.ignore_memlock_limit:
        memlock_errors = tuple(
            item for item in preflight.errors if "RLIMIT_MEMLOCK" in item
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
                    "explicitly ignored after user confirmation: " + item
                    for item in memlock_errors
                ),
            )
    preflight.raise_for_error()
    print(preflight.format_text(), file=sys.stderr)

    with ExitStack() as resources:
        store = resources.enter_context(
            MultiDtypeWeightStore(
                plan,
                args.weight_store,
                staging_slot_count=args.slots,
            )
        )
        load_started = time.perf_counter()
        store.load_checkpoint(args.checkpoint)
        load_seconds = time.perf_counter() - load_started
        resident = resources.enter_context(
            MixedResidentDeviceArena(
                plan,
                store,
                device,
                transformer_placement=transformer_placement,
            )
        )
        runtime = resources.enter_context(
            MixedDtypeRuntime(
                plan,
                store,
                resident,
                device,
                slot_count=args.slots,
                prefetch_depth=prefetch_depth,
                profile=args.profile,
            )
        )
        if backend_phase_plan is not None:
            runtime.configure_backend_phase_plan(backend_phase_plan)
        vocab_runtime = None
        if plan.vocab.stream_embedding or plan.vocab.stream_lm_head:
            vocab_runtime = resources.enter_context(MixedVocabStreamingRuntime(
                plan,
                store,
                runtime,
                embedding_staging_rows=max(1, int(encoded.input_ids.numel())),
                profile=args.profile,
            ))
        kv_dtype = (
            torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16
        )
        executor = resources.enter_context(Llama31DecodeExecutor(
            config,
            resident,
            vocab_runtime=vocab_runtime,
            top_k=args.top_k,
            return_full_logits=args.return_full_logits,
            max_cache_length=cache_length,
            kv_block_size=args.kv_block_size,
            kv_dtype=kv_dtype,
            kv_policy=kv_policy,
        ))
        input_ids = encoded.input_ids.to(device)
        generated = []
        token_latencies = []
        runtime_profiles = []
        vocab_profiles = []

        def capture_profiles():
            runtime_profiles.append(dict(runtime.last_profile or {}))
            if vocab_runtime is not None:
                vocab_profiles.append(vocab_runtime.profile_stats())

        generation_started = time.perf_counter()
        with torch.inference_mode():
            prefill_started = time.perf_counter()
            state = executor.begin(input_ids)
            state = runtime.run(executor, state)
            state = executor.finish(state)
            torch.cuda.synchronize(device)
            time_to_first_token_ms = (
                time.perf_counter() - prefill_started
            ) * 1000.0
            capture_profiles()
            next_token = state.topk_indices[..., 0]
            generated.append(next_token)

            for _ in range(args.max_new_tokens - 1):
                started = time.perf_counter()
                state = executor.begin(next_token)
                state = runtime.run(executor, state)
                state = executor.finish(state)
                next_token = state.topk_indices[..., 0]
                torch.cuda.synchronize(device)
                token_latencies.append(
                    (time.perf_counter() - started) * 1000.0
                )
                capture_profiles()
                generated.append(next_token)
        generation_wall_seconds = time.perf_counter() - generation_started

        generated_ids = torch.cat(generated, dim=-1).cpu()
        text_output = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        vocab_pinned_bytes = (
                vocab_runtime.extra_pinned_cpu_bytes
                if vocab_runtime is not None
                else 0
        )
        report = build_inference_report(
            plan=plan,
            geometry=geometry,
            policy=policy,
            checkpoint=args.checkpoint,
            validation=validation,
            preflight=preflight,
            checkpoint_load_seconds=load_seconds,
            prompt_tokens=int(input_ids.numel()),
            generated_tokens=args.max_new_tokens,
            generation_wall_seconds=generation_wall_seconds,
            time_to_first_token_ms=time_to_first_token_ms,
            decode_token_latencies_ms=token_latencies,
            runtime_profiles=runtime_profiles,
            vocab_profiles=vocab_profiles,
            gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(device),
            cpu_resident_bytes=plan.host_arena_bytes,
            pinned_bytes=store.pinned_bytes + vocab_pinned_bytes,
            kv_cache_bytes=executor.kv_cache.nbytes,
            kv_profiles=[executor.kv_cache.profile_stats()],
            device=device,
        )
        result = report.as_dict()
        if compatibility_report is not None:
            result["hardware"]["compatibility"] = (
                compatibility_report.as_dict()
            )
        result["transformer_placement"] = transformer_placement.as_dict()
        result["generation"] = {
            "prompt": args.prompt,
            "generated_text": text_output,
            "generated_token_ids": generated_ids[0].tolist(),
            "top_k": args.top_k,
            "return_full_logits": args.return_full_logits,
        }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
