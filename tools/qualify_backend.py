#!/usr/bin/env python3
"""Qualify a linear backend on decode/prefill Llama matrix shapes.

Synthetic results are useful for provider eligibility and regression work; they
are not reported as real-model performance.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming import (  # noqa: E402
    QuantizationSpec,
    WeightSpec,
    backend_for_weight,
    pack_int4,
    qualify_backend,
)


MODEL_SHAPES = {
    "tiny": (256, 768),
    "8b": (4096, 14336),
    "70b": (8192, 28672),
}
DEFAULT_M_VALUES = (1, 2, 4, 8, 16, 32, 128, 512)
MINIMUM_TOPK_CONSISTENCY = 0.9


def parser():
    result = argparse.ArgumentParser(
        description=(
            "Qualify one explicit backend; no fallback is selected implicitly."
        )
    )
    result.add_argument("--backend", required=True)
    result.add_argument(
        "--provider",
        choices=("none", "cutlass"),
        default="none",
        help="Explicitly load an optional provider before qualification.",
    )
    result.add_argument("--provider-library")
    result.add_argument(
        "--preset",
        choices=("tiny", "8b", "70b", "all"),
        default="tiny",
    )
    result.add_argument(
        "--matrix",
        choices=("hidden_hidden", "up", "down", "all"),
        default="all",
    )
    result.add_argument("--m", type=int, nargs="+", default=DEFAULT_M_VALUES)
    result.add_argument("--bits", type=int, choices=(4, 8), default=8)
    result.add_argument(
        "--granularity",
        choices=("per_channel", "per_group"),
        default="per_channel",
    )
    result.add_argument("--group-size", type=int, default=64)
    result.add_argument(
        "--activation-dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    result.add_argument(
        "--scale-dtype", choices=("bf16", "fp16"), default="bf16"
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument("--iterations", type=int, default=10)
    result.add_argument(
        "--metadata-only",
        action="store_true",
        help="Run static capability checks without allocating tensors.",
    )
    result.add_argument("--output")
    return result


def torch_dtype(name):
    return torch.bfloat16 if name == "bf16" else torch.float16


def cases(args):
    presets = MODEL_SHAPES if args.preset == "all" else {
        args.preset: MODEL_SHAPES[args.preset]
    }
    for preset, (hidden, intermediate) in presets.items():
        matrices = {
            "hidden_hidden": (hidden, hidden),
            "up": (intermediate, hidden),
            "down": (hidden, intermediate),
        }
        selected = matrices if args.matrix == "all" else {
            args.matrix: matrices[args.matrix]
        }
        for matrix, (n, k) in selected.items():
            for m in args.m:
                yield preset, matrix, int(m), int(n), int(k)


def make_spec(args, n, k):
    quant = QuantizationSpec(
        bits=args.bits,
        scheme="symmetric",
        granularity=args.granularity,
        group_size=(
            args.group_size if args.granularity == "per_group" else None
        ),
        scale_dtype=args.scale_dtype,
        packing="int4_pair_uint8" if args.bits == 4 else None,
        axis=1,
    )
    return WeightSpec.quantized(
        "qualification.weight",
        (n, k),
        quant,
        args.activation_dtype,
        "attention_q",
    )


def allocate_inputs(args, spec, m, device):
    torch.manual_seed(1234 + int(m) + int(spec.logical_shape[0]))
    activation_dtype = torch_dtype(args.activation_dtype)
    scale_dtype = torch_dtype(args.scale_dtype)
    n, k = spec.logical_shape
    x = torch.randn((m, k), dtype=activation_dtype, device=device)
    low, high = (-8, 7) if args.bits == 4 else (-127, 127)
    quantized = torch.randint(
        low,
        high + 1,
        (n, k),
        dtype=torch.int8,
        device=device,
    )
    stored = pack_int4(quantized) if args.bits == 4 else quantized
    scale_shape = (
        (n, 1)
        if args.granularity == "per_channel"
        else (n, k // args.group_size)
    )
    scale = (
        torch.rand(scale_shape, dtype=scale_dtype, device=device) * 0.02
        + 0.001
    )
    expanded = (
        scale
        if args.granularity == "per_channel"
        else scale.repeat_interleave(args.group_size, dim=1)
    )
    reference_weight = quantized.to(activation_dtype) * expanded.to(
        activation_dtype
    )
    return x, stored, scale, reference_weight


def error_metrics(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    expected_mean_abs = expected.float().abs().mean()
    k = min(10, actual.shape[-1])
    actual_topk = torch.topk(actual.float(), k, dim=-1).indices
    expected_topk = torch.topk(expected.float(), k, dim=-1).indices
    intersection = 0
    top1_matches = 0
    for left, right in zip(actual_topk, expected_topk):
        intersection += len(set(left.tolist()).intersection(right.tolist()))
        top1_matches += int(left[0].item() == right[0].item())
    return {
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "relative_error": float(
            (difference.mean() / expected_mean_abs.clamp_min(1e-6)).item()
        ),
        "expected_mean_abs": float(expected_mean_abs.item()),
        "topk_consistency": intersection / float(actual_topk.numel()),
        "top1_consistency": top1_matches / float(actual_topk.shape[0]),
    }


def run_case(args, preset, matrix, m, n, k):
    spec = make_spec(args, n, k)
    qualification = qualify_backend(
        spec, args.backend, m, device=args.device
    )
    result = {
        "preset": preset,
        "matrix": matrix,
        **qualification.as_dict(),
        "numerical_error": None,
        "latency_ms": None,
    }
    if args.metadata_only or not qualification.supported:
        return result
    device = torch.device(args.device)
    backend = backend_for_weight(spec, backend_name=args.backend)
    try:
        x, stored, scale, reference_weight = allocate_inputs(
            args, spec, m, device
        )
        workspace_bytes = backend.workspace_bytes(spec, batch_tokens=m)
        workspace_storage = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
        workspace = workspace_storage.view(
            torch_dtype(args.activation_dtype)
        )
        quant_views = {"scale": scale, "weight_spec": spec}
        with torch.inference_mode():
            expected = F.linear(x, reference_weight)
            for _ in range(args.warmup):
                backend.execute(x, stored, quant_views, workspace)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            actual = None
            for _ in range(args.iterations):
                actual = backend.execute(x, stored, quant_views, workspace)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
        result["latency_ms"] = {
            "cuda_event_median_proxy": start.elapsed_time(end)
            / args.iterations,
            "wall_mean": wall_ms / args.iterations,
            "iterations": args.iterations,
        }
        result["numerical_error"] = error_metrics(actual, expected)
        tolerance = backend.info
        result["numerical_tolerance"] = {
            "atol": float(tolerance.atol),
            "rtol": float(tolerance.rtol),
            "mean_relative_error": float(tolerance.rtol),
            "mean_error_bound": float(
                tolerance.atol
                + tolerance.rtol
                * result["numerical_error"]["expected_mean_abs"]
            ),
            "minimum_topk_consistency": MINIMUM_TOPK_CONSISTENCY,
        }
        result["numerical_pass"] = bool(
            result["numerical_error"]["mean_abs_error"]
            <= result["numerical_tolerance"]["mean_error_bound"]
            and result["numerical_error"]["relative_error"]
            <= tolerance.rtol
            and result["numerical_error"]["topk_consistency"]
            >= MINIMUM_TOPK_CONSISTENCY
        )
        if not result["numerical_pass"]:
            result["supported"] = False
            result["unsupported_reason"] = (
                "numerical result exceeds backend tolerance"
            )
    except BaseException as error:
        result["supported"] = False
        result["unsupported_reason"] = "{}: {}".format(
            type(error).__name__, str(error).splitlines()[0]
        )
    return result


def main():
    args = parser().parse_args()
    if args.provider == "cutlass":
        from layer_streaming.providers.cutlass import (
            load_cutlass_w8a16_provider,
        )

        load_cutlass_w8a16_provider(args.provider_library)
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("warmup must be >= 0 and iterations must be positive")
    if any(item < 1 for item in args.m):
        raise SystemExit("all M values must be positive")
    if (
        args.granularity == "per_group"
        and any(k % args.group_size for _, _, _, _, k in cases(args))
    ):
        raise SystemExit("group size must divide every selected K")
    payload = {
        "schema_version": 1,
        "synthetic_only": True,
        "backend": args.backend,
        "device": args.device,
        "configuration": {
            "preset": args.preset,
            "matrix": args.matrix,
            "m": args.m,
            "bits": args.bits,
            "granularity": args.granularity,
            "group_size": (
                args.group_size
                if args.granularity == "per_group"
                else None
            ),
            "activation_dtype": args.activation_dtype,
            "scale_dtype": args.scale_dtype,
            "metadata_only": args.metadata_only,
        },
        "results": [run_case(args, *case) for case in cases(args)],
    }
    payload["supported"] = all(
        item["supported"] for item in payload["results"]
    )
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if payload["supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
