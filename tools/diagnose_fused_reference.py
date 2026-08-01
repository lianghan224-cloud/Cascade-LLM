#!/usr/bin/env python3
"""Diagnose fused W8A16 against explicit dequantization on identical inputs."""

import argparse
from argparse import Namespace
import json
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_reference import (  # noqa: E402
    compare,
    parse_byte_size,
    parse_ids,
    streaming_outputs,
)
from layer_streaming.providers.cutlass import (  # noqa: E402
    load_cutlass_w8a16_provider,
)


MATRIX_PATTERN = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.(?:self_attn|mlp)\."
    r"(?P<matrix>[^.]+)\.weight"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run fallback prefill followed by fused decode and compare every "
            "fused linear with explicit BF16 dequantization on the same input."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--provider-library")
    parser.add_argument(
        "--input-ids", type=parse_ids, default=[128000, 128006, 882, 220]
    )
    parser.add_argument(
        "--decode-ids", type=parse_ids, default=[128001, 128001]
    )
    parser.add_argument("--device", default="cuda:0")
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
    parser.add_argument("--slots", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--gpu-resident-weight-budget", type=parse_byte_size, default=0
    )
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.04)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--bf16-reduction-mode",
        choices=("default", "fp32"),
        default="default",
        help=(
            "Use PyTorch's default BF16 reduction or disable reduced-precision "
            "reduction while constructing the explicit fallback reference."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_stats(tensor):
    value = tensor.detach().float()
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "mean_abs": float(value.abs().mean().item()),
    }


def error_metrics(actual, expected, atol, rtol, top_k=10):
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = (actual_float - expected_float).abs()
    expected_abs = expected_float.abs()
    expected_mean_abs = expected_abs.mean().clamp_min(1.0e-8)
    threshold = float(atol) + float(rtol) * expected_abs
    count = min(int(top_k), int(actual.shape[-1]))
    actual_topk = torch.topk(actual_float.reshape(-1, actual.shape[-1]), count).indices
    expected_topk = torch.topk(
        expected_float.reshape(-1, expected.shape[-1]), count
    ).indices
    intersection = (
        actual_topk.unsqueeze(-1) == expected_topk.unsqueeze(-2)
    ).any(dim=-1).sum().item()
    max_flat = int(difference.argmax().item())
    max_index = list(torch.unravel_index(
        torch.tensor(max_flat, device=difference.device), difference.shape
    ))
    max_index = [int(item.item()) for item in max_index]
    return {
        "allclose": bool(
            torch.allclose(actual_float, expected_float, atol=atol, rtol=rtol)
        ),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "mean_relative_error": float(
            (difference.mean() / expected_mean_abs).item()
        ),
        "max_relative_error_clamped": float(
            (difference / expected_abs.clamp_min(1.0e-3)).max().item()
        ),
        "expected_mean_abs": float(expected_mean_abs.item()),
        "elements_over_allclose_bound": int((difference > threshold).sum().item()),
        "element_count": int(difference.numel()),
        "top1_equal": bool(torch.equal(actual_topk[..., :1], expected_topk[..., :1])),
        "topk_consistency": float(intersection) / float(actual_topk.numel()),
        "max_error_index": max_index,
        "actual_at_max_error": float(actual_float[tuple(max_index)].item()),
        "expected_at_max_error": float(expected_float[tuple(max_index)].item()),
    }


def dequantized_weight(weight):
    spec = weight.quant_views["weight_spec"]
    raw = weight.weight_view
    scale = weight.quant_views["scale"]
    quant = spec.quantization
    expanded = (
        scale
        if quant.granularity == "per_channel"
        else scale.repeat_interleave(int(quant.group_size), dim=1)
    )
    return raw.to(torch_dtype(spec.compute_dtype)) * expanded.to(
        torch_dtype(spec.compute_dtype)
    )


def torch_dtype(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(name)]


class LinearDiagnosticCollector:
    def __init__(self, atol, rtol, top_k):
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.top_k = int(top_k)
        self.occurrences = {}
        self.records = []

    def __call__(self, name, inputs, weight, actual):
        # Prefill uses the explicit fallback and receives a dense Tensor.
        # Only fused weights expose the ephemeral raw/scale views below.
        if not hasattr(weight, "quant_views"):
            return
        occurrence = self.occurrences.get(name, 0)
        self.occurrences[name] = occurrence + 1
        match = MATRIX_PATTERN.fullmatch(name)
        if match is None:
            raise RuntimeError("cannot parse Transformer matrix name {}".format(name))
        spec = weight.quant_views["weight_spec"]
        raw = weight.weight_view
        scale = weight.quant_views["scale"]
        with torch.inference_mode():
            explicit_weight = dequantized_weight(weight)
            fallback = F.linear(inputs.to(explicit_weight.dtype), explicit_weight)
            fp32 = F.linear(
                inputs.float(), explicit_weight.float()
            ).to(inputs.dtype)
            raw_output = F.linear(inputs, raw.to(inputs.dtype))
            if spec.quantization.granularity == "per_channel":
                post_scale = (
                    raw_output * scale.reshape(
                        *((1,) * (raw_output.ndim - 1)), -1
                    ).to(raw_output.dtype)
                )
            else:
                post_scale = None
            record = {
                "phase": "decode_{}".format(occurrence),
                "layer": int(match.group("layer")),
                "matrix": match.group("matrix"),
                "weight_name": name,
                "input": tensor_stats(inputs),
                "output": tensor_stats(actual),
                "weight": {
                    "storage_dtype": spec.storage_dtype,
                    "compute_dtype": spec.compute_dtype,
                    "logical_shape": list(spec.logical_shape),
                    "storage_shape": list(spec.storage_shape),
                    "storage_min": int(raw.min().item()),
                    "storage_max": int(raw.max().item()),
                    "quantization": spec.quantization.as_dict(),
                    "k_aligned_16": int(spec.logical_shape[1]) % 16 == 0,
                    "n_aligned_8": int(spec.logical_shape[0]) % 8 == 0,
                    "has_storage_padding": (
                        tuple(spec.logical_shape) != tuple(spec.storage_shape)
                    ),
                },
                "scale": tensor_stats(scale),
                "fused_vs_fallback": error_metrics(
                    actual, fallback, self.atol, self.rtol, self.top_k
                ),
                "fused_vs_fp32_accumulation": error_metrics(
                    actual, fp32, self.atol, self.rtol, self.top_k
                ),
                "fallback_vs_fp32_accumulation": error_metrics(
                    fallback, fp32, self.atol, self.rtol, self.top_k
                ),
                "post_scale_vs_fallback": (
                    None
                    if post_scale is None
                    else error_metrics(
                        post_scale,
                        fallback,
                        self.atol,
                        self.rtol,
                        self.top_k,
                    )
                ),
            }
            self.records.append(record)
            del explicit_weight, fallback, fp32, raw_output, post_scale


def runtime_args(args, linear_trace_callback=None, fused_decode=False):
    return Namespace(
        checkpoint=args.checkpoint,
        weight_format="auto",
        backend="int8_dequant_bf16_fallback",
        prefill_backend="int8_dequant_bf16_fallback",
        decode_backend=(
            "fused_w8a16"
            if fused_decode
            else "int8_dequant_bf16_fallback"
        ),
        granularity=args.granularity,
        weight_store=args.weight_store,
        embedding_mode="streamed",
        lm_head_mode="streamed",
        slots=args.slots,
        block_size=16,
        top_k=args.top_k,
        profile=False,
        gpu_resident_weight_budget=args.gpu_resident_weight_budget,
        linear_trace_callback=linear_trace_callback,
    )


def propagation_order(name):
    phase, stage, *tail = name.split("/")
    phase_index = int(phase.split("_")[1]) if phase.startswith("decode_") else -1
    if stage == "embedding":
        return phase_index, -1, 0
    if stage in {"attention", "mlp", "hidden"}:
        layer = int(tail[0])
        stage_order = {"attention": 0, "mlp": 1, "hidden": 2}[stage]
        return phase_index, layer, stage_order
    return phase_index, 10 ** 6, {"final_norm": 0, "logits": 1}.get(stage, 2)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    provider = load_cutlass_w8a16_provider(args.provider_library)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    device = torch.device(args.device)
    prefill = torch.tensor([args.input_ids], dtype=torch.long, device=device)
    decode = [
        torch.tensor([[token]], dtype=torch.long, device=device)
        for token in args.decode_ids
    ]
    original_reduction = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    if args.bf16_reduction_mode == "fp32":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    collector = LinearDiagnosticCollector(args.atol, args.rtol, args.top_k)
    try:
        reference = streaming_outputs(
            runtime_args(args), config, device, prefill, decode
        )
        candidate = streaming_outputs(
            runtime_args(args, collector, fused_decode=True),
            config,
            device,
            prefill,
            decode,
        )
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            original_reduction
        )
    _, comparisons = compare(
        reference, candidate, args.atol, args.rtol, args.top_k
    )
    propagation = [
        {"stage": name, **value}
        for name, value in sorted(
            comparisons.items(), key=lambda item: propagation_order(item[0])
        )
        if name.startswith("decode_") and value.get("max_abs_error", 0.0) > 0.0
    ]
    failed_stages = [
        item for item in propagation if not item.get("ok", False)
    ]
    local_failures = [
        item
        for item in collector.records
        if not item["fused_vs_fallback"]["allclose"]
    ]
    worst_relative = max(
        collector.records,
        key=lambda item: item["fused_vs_fallback"]["mean_relative_error"],
    )
    worst_absolute = max(
        collector.records,
        key=lambda item: item["fused_vs_fallback"]["max_abs_error"],
    )
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "provider": "cutlass_sm86_w8a16",
        "provider_abi": provider.provider_version,
        "model_signature": {
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
        },
        "configuration": {
            "input_ids": args.input_ids,
            "decode_ids": args.decode_ids,
            "reference_prefill_backend": "int8_dequant_bf16_fallback",
            "reference_decode_backend": "int8_dequant_bf16_fallback",
            "candidate_prefill_backend": "int8_dequant_bf16_fallback",
            "candidate_decode_backend": "fused_w8a16",
            "granularity": args.granularity,
            "weight_store": args.weight_store,
            "gpu_resident_weight_budget_bytes": args.gpu_resident_weight_budget,
            "atol": args.atol,
            "rtol": args.rtol,
            "top_k": args.top_k,
            "bf16_reduction_mode": args.bf16_reduction_mode,
        },
        "summary": {
            "stage_comparison_count": len(comparisons),
            "failed_stage_count": len(failed_stages),
            "linear_comparison_count": len(collector.records),
            "local_linear_allclose_failure_count": len(local_failures),
            "worst_local_relative": {
                "phase": worst_relative["phase"],
                "layer": worst_relative["layer"],
                "matrix": worst_relative["matrix"],
                **worst_relative["fused_vs_fallback"],
            },
            "worst_local_absolute": {
                "phase": worst_absolute["phase"],
                "layer": worst_absolute["layer"],
                "matrix": worst_absolute["matrix"],
                **worst_absolute["fused_vs_fallback"],
            },
        },
        "failed_stages": failed_stages,
        "error_propagation": propagation,
        "local_linear_failures": local_failures,
        "linear_diagnostics": collector.records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        **report["summary"],
        "failed_stages": [item["stage"] for item in failed_stages],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
