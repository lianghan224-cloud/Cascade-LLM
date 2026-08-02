#!/usr/bin/env python3
"""Compare streamed BF16 stages with Hugging Face Llama outputs."""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, LlamaForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    BackendSelection,
    CheckpointManifest,
    ExecutionPolicy,
    KVPolicy,
    Llama31DecodeExecutor,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MixedVocabStreamingRuntime,
    MultiDtypeWeightStore,
    PlacementMode,
    WeightFormat,
    adapter_for_config,
    backend_for_weight,
    build_backend_phase_plan,
    build_static_transformer_placement,
)


def parse_ids(value):
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("token list cannot be empty")
    return result


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
    parser.add_argument("--input-ids", type=parse_ids, default=[1, 17, 42, 9])
    parser.add_argument("--decode-ids", type=parse_ids, default=[23])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reference-device-map",
        choices=("single", "balanced"),
        default="single",
        help=(
            "Load the Hugging Face reference on --device or distribute it "
            "across all visible GPUs. Balanced mode is intended for real "
            "checkpoints that do not fit on one GPU."
        ),
    )
    parser.add_argument(
        "--weight-format",
        choices=("auto",) + tuple(item.value for item in WeightFormat),
        default="auto",
    )
    parser.add_argument(
        "--backend",
        choices=(
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
        ),
        default="checkpoint",
        help="Explicit compute backend; unavailable fused choices fail early.",
    )
    parser.add_argument(
        "--provider", choices=("none", "cutlass"), default="none"
    )
    parser.add_argument("--provider-library")
    parser.add_argument(
        "--granularity",
        choices=("matrix", "matrix_group", "layer"),
        default="matrix_group",
    )
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="pinned_staging",
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
    parser.add_argument("--slots", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--block-size", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--kv-attention-backend",
        choices=(
            "generic_cuda",
            "sm80",
            "sm86",
            "sm89",
            "sm90",
            "reference_paged_exact",
            "legacy_gather_sdpa_reference",
        ),
        default="generic_cuda",
    )
    parser.add_argument(
        "--allow-kv-reference",
        action="store_true",
        help="Explicitly permit a diagnostic reference paged provider.",
    )
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--gpu-resident-weight-budget",
        type=parse_byte_size,
        default=0,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def cpu_trace(tensor):
    return tensor.detach().float().cpu()


def load_checkpoint_tensors(checkpoint):
    checkpoint = Path(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files = sorted(set(index["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    tensors = {}
    for relative_name in files:
        with safe_open(
            str(checkpoint / relative_name), framework="pt", device="cpu"
        ) as source:
            for key in source.keys():
                tensors[key] = source.get_tensor(key)
    return tensors


def explicit_reference_model(checkpoint, config, policy):
    tensors = load_checkpoint_tensors(checkpoint)
    adapter = adapter_for_config(config)
    plan = adapter.build_execution_plan(config, policy)
    compute_dtype = next(
        spec.compute_dtype
        for spec in plan.weights.values()
        if spec.role == "attention_q"
    )
    torch_dtype = (
        torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16
    )
    model = LlamaForCausalLM(config).to(torch_dtype).eval()
    reference_state = {}
    for key in model.state_dict():
        spec = plan.weights[key]
        storage_key = spec.alias_of or key
        storage_spec = plan.weights[storage_key]
        source = tensors[storage_key]
        if storage_spec.quantization is not None:
            source = backend_for_weight(storage_spec).dequantize(
                storage_spec,
                source,
                {
                    "scale": tensors[storage_key + "_scale"],
                    "weight_spec": storage_spec,
                },
            )
        reference_state[key] = source.to(torch_dtype)
    model.load_state_dict(reference_state, strict=True)
    return model


def reference_outputs(
    checkpoint,
    config,
    device,
    prefill_ids,
    decode_ids,
    policy,
    device_map="single",
):
    if device_map == "single":
        model = explicit_reference_model(checkpoint, config, policy)
        model = model.to(device).eval()
    else:
        if policy.quantization is not None:
            raise ValueError(
                "balanced Hugging Face reference loading currently supports "
                "unquantized checkpoints only"
            )
        compute_dtype = (
            torch.bfloat16
            if policy.weight_format == WeightFormat.BF16
            else torch.float16
        )
        model = LlamaForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=compute_dtype,
            device_map="balanced",
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
    input_device = model.model.embed_tokens.weight.device
    prefill_ids = prefill_ids.to(input_device)
    decode_ids = [item.to(input_device) for item in decode_ids]
    traces = {}
    phase = ["prefill"]
    hooks = []

    def capture(name, layer=None):
        def hook(module, inputs, output):
            del module, inputs
            tensor = output[0] if isinstance(output, tuple) else output
            key = "{}/{}".format(
                phase[0], name if layer is None else "{}/{}".format(name, layer)
            )
            traces[key] = cpu_trace(tensor)

        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(capture("embedding")))
    for index, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(capture("attention", index)))
        hooks.append(layer.mlp.register_forward_hook(capture("mlp", index)))
        hooks.append(layer.register_forward_hook(capture("hidden", index)))
    hooks.append(model.model.norm.register_forward_hook(capture("final_norm")))
    with torch.inference_mode():
        output = model(prefill_ids, use_cache=True)
        traces["prefill/logits"] = cpu_trace(output.logits)
        cache = output.past_key_values
        for index, token in enumerate(decode_ids):
            phase[0] = "decode_{}".format(index)
            output = model(token, past_key_values=cache, use_cache=True)
            traces[phase[0] + "/logits"] = cpu_trace(output.logits)
            cache = output.past_key_values
    for handle in hooks:
        handle.remove()
    del hooks, handle, model, output, cache
    gc.collect()
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
    return traces


def streaming_outputs(args, config, device, prefill_ids, decode_ids):
    inferred = ExecutionPolicy.from_config(config)
    if args.weight_format != "auto" and WeightFormat(args.weight_format) != inferred.weight_format:
        raise ValueError(
            "--weight-format {} conflicts with checkpoint metadata {}".format(
                args.weight_format, inferred.weight_format.value
            )
        )
    policy = ExecutionPolicy(
        granularity=args.granularity,
        weight_format=inferred.weight_format,
        cpu_weight_mode=args.weight_store,
        embedding_mode=PlacementMode(args.embedding_mode),
        lm_head_mode=PlacementMode(args.lm_head_mode),
        slot_count=args.slots,
        prefetch_depth=args.slots,
        embedding_dtype=inferred.embedding_dtype,
        lm_head_dtype=inferred.lm_head_dtype,
        norm_dtype=inferred.norm_dtype,
        quantization=inferred.quantization,
        linear_backend=(
            None if args.backend == "checkpoint" else args.backend
        ),
    )
    adapter = adapter_for_config(config)
    plan = adapter.build_execution_plan(config, policy)
    placement = build_static_transformer_placement(
        plan, args.gpu_resident_weight_budget
    )
    planned_backends = {
        tensor.backend
        for unit in plan.units
        for tensor in unit.tensors
        if tensor.backend
    }
    if args.backend != "checkpoint" and planned_backends != {args.backend}:
        raise ValueError(
            "requested backend {}, plan uses {}".format(
                args.backend, sorted(planned_backends)
            )
        )
    if len(planned_backends) != 1:
        raise ValueError(
            "phase qualification requires one Transformer backend, got {}".format(
                sorted(planned_backends)
            )
        )
    actual_backend = next(iter(planned_backends))
    prefill_backend = getattr(args, "prefill_backend", None) or actual_backend
    decode_backend = getattr(args, "decode_backend", None) or actual_backend
    backend_phase_plan = build_backend_phase_plan(
        plan,
        BackendSelection(
            prefill=prefill_backend,
            decode=decode_backend,
        ),
        device=device,
        prefill_m_values=(int(prefill_ids.numel()),),
        decode_m_values=(1,),
    )
    adapter.validate_checkpoint(args.checkpoint, config, policy).raise_for_error()
    traces = {}
    phase = ["prefill"]

    def capture(stage, layer, tensor):
        name = stage if layer is None else "{}/{}".format(stage, layer)
        if stage in {"embedding", "attention", "mlp", "hidden", "final_norm", "logits"}:
            traces["{}/{}".format(phase[0], name)] = cpu_trace(tensor)

    max_length = prefill_ids.shape[1] + len(decode_ids)
    with ExitStack() as resources:
        store = resources.enter_context(
            MultiDtypeWeightStore(
                plan,
                args.weight_store,
                staging_slot_count=args.slots,
            )
        )
        store.load_checkpoint(args.checkpoint)
        resident = resources.enter_context(
            MixedResidentDeviceArena(
                plan,
                store,
                device,
                transformer_placement=placement,
            )
        )
        runtime = resources.enter_context(
            MixedDtypeRuntime(
                plan,
                store,
                resident,
                device,
                slot_count=args.slots,
                profile=args.profile,
            )
        )
        runtime.configure_backend_phase_plan(backend_phase_plan)
        vocab = None
        if plan.vocab.stream_embedding or plan.vocab.stream_lm_head:
            vocab = resources.enter_context(MixedVocabStreamingRuntime(
                plan,
                store,
                runtime,
                embedding_staging_rows=max_length,
                profile=args.profile,
            ))
        executor = resources.enter_context(Llama31DecodeExecutor(
            config,
            resident,
            vocab_runtime=vocab,
            top_k=args.top_k,
            return_full_logits=True,
            max_cache_length=max_length,
            kv_block_size=args.block_size,
            kv_policy=KVPolicy(
                attention_backend=args.kv_attention_backend,
                page_size=args.block_size,
                dtype=(
                    "bf16"
                    if next(
                        spec.compute_dtype
                        for spec in plan.weights.values()
                        if spec.role == "attention_q"
                    )
                    == "bfloat16"
                    else "fp16"
                ),
            ),
            allow_kv_reference=args.allow_kv_reference,
            trace_callback=capture,
            linear_trace_callback=getattr(
                args, "linear_trace_callback", None
            ),
            kv_dtype=(
                torch.bfloat16
                if next(
                    spec.compute_dtype
                    for spec in plan.weights.values()
                    if spec.role == "attention_q"
                )
                == "bfloat16"
                else torch.float16
            ),
        ))
        with torch.inference_mode():
            state = executor.finish(runtime.run(executor, executor.begin(prefill_ids)))
            traces["prefill/logits"] = cpu_trace(state.logits)
            for index, token in enumerate(decode_ids):
                phase[0] = "decode_{}".format(index)
                state = executor.finish(runtime.run(executor, executor.begin(token)))
                traces[phase[0] + "/logits"] = cpu_trace(state.logits)
    return traces


def compare(reference, candidate, atol, rtol, top_k):
    results = {}
    all_ok = True
    for key in sorted(set(reference).union(candidate)):
        expected = reference.get(key)
        actual = candidate.get(key)
        if expected is None or actual is None:
            results[key] = {"ok": False, "reason": "missing output"}
            all_ok = False
            continue
        difference = (expected - actual).abs()
        relative = difference / expected.abs().clamp_min(1.0e-8)
        elementwise_ok = bool(
            torch.allclose(expected, actual, atol=atol, rtol=rtol)
        )
        item = {
            "ok": elementwise_ok,
            "elementwise_ok": elementwise_ok,
            "shape": list(expected.shape),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "max_relative_error": float(relative.max().item()),
            "mean_relative_error": float(relative.mean().item()),
        }
        if key.endswith("/logits"):
            count = min(int(top_k), expected.shape[-1])
            expected_topk = torch.topk(expected[:, -1:, :], count, dim=-1).indices
            actual_topk = torch.topk(actual[:, -1:, :], count, dim=-1).indices
            intersection = (
                expected_topk.unsqueeze(-1)
                == actual_topk.unsqueeze(-2)
            ).any(dim=-1).sum().item()
            item["topk_consistency"] = float(intersection) / float(
                expected_topk.numel()
            )
            item["topk_equal"] = bool(torch.equal(expected_topk, actual_topk))
            item["top1_equal"] = bool(
                torch.equal(expected_topk[..., :1], actual_topk[..., :1])
            )
            item["ok"] = item["ok"] and item["topk_equal"]
        all_ok = all_ok and item["ok"]
        results[key] = item
    return all_ok, results


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.provider == "cutlass":
        from layer_streaming.providers.cutlass import (
            load_cutlass_w8a16_provider,
        )

        load_cutlass_w8a16_provider(args.provider_library)
    torch.manual_seed(0)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    inferred_policy = ExecutionPolicy.from_config(config)
    if (
        args.weight_format != "auto"
        and WeightFormat(args.weight_format) != inferred_policy.weight_format
    ):
        raise SystemExit(
            "--weight-format conflicts with checkpoint quantization metadata"
        )
    prefill = torch.tensor([args.input_ids], dtype=torch.long, device=device)
    decode = [
        torch.tensor([[token]], dtype=torch.long, device=device)
        for token in args.decode_ids
    ]
    reference = reference_outputs(
        args.checkpoint,
        config,
        device,
        prefill,
        decode,
        inferred_policy,
        device_map=args.reference_device_map,
    )
    candidate = streaming_outputs(args, config, device, prefill, decode)
    ok, comparisons = compare(
        reference, candidate, args.atol, args.rtol, args.top_k
    )
    logits_comparisons = [
        value
        for key, value in comparisons.items()
        if key.endswith("/logits")
    ]
    report = {
        "ok": ok,
        "checkpoint": str(args.checkpoint),
        "weight_format": inferred_policy.weight_format.value,
        "backend_requested": args.backend,
        "kv_attention_backend": args.kv_attention_backend,
        "lm_head_backend": "deterministic_cuda_fp32_accum_native_output",
        "reference_device_map": args.reference_device_map,
        "gpu_resident_weight_budget_bytes": (
            args.gpu_resident_weight_budget
        ),
        "input_ids": args.input_ids,
        "decode_ids": args.decode_ids,
        "atol": args.atol,
        "rtol": args.rtol,
        "acceptance": {
            "strict_hf_elementwise_and_ordered_topk": bool(ok),
            "greedy_token_all_equal": bool(logits_comparisons) and all(
                item.get("top1_equal", False) for item in logits_comparisons
            ),
            "minimum_topk_set_consistency": (
                min(item["topk_consistency"] for item in logits_comparisons)
                if logits_comparisons
                else None
            ),
            "note": (
                "HF elementwise gate remains diagnostic for alternative "
                "paged-attention reduction paths; it is not rewritten when "
                "the architecture-specific kernel contract passes."
            ),
        },
        "comparisons": comparisons,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
