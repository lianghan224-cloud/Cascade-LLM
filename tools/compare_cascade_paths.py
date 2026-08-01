#!/usr/bin/env python3
"""Compare two Cascade checkpoint/backend paths stage by stage."""

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_reference import compare, parse_byte_size, parse_ids, streaming_outputs  # noqa: E402
from layer_streaming import adapter_for_config  # noqa: E402


BACKENDS = (
    "checkpoint",
    "bf16_linear",
    "fp16_linear",
    "int8_dequant_bf16_fallback",
    "int8_dequant_fp16_fallback",
    "fused_w8a16",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-backend", choices=BACKENDS, required=True)
    parser.add_argument("--candidate-backend", choices=BACKENDS, required=True)
    parser.add_argument("--reference-prefill-backend", choices=BACKENDS)
    parser.add_argument("--reference-decode-backend", choices=BACKENDS)
    parser.add_argument("--candidate-prefill-backend", choices=BACKENDS)
    parser.add_argument("--candidate-decode-backend", choices=BACKENDS)
    parser.add_argument("--provider", choices=("none", "cutlass"), default="none")
    parser.add_argument("--provider-library")
    parser.add_argument("--input-ids", type=parse_ids, default=[128000, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--decode-ids", type=parse_ids, default=[4, 5])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--granularity", choices=("matrix", "matrix_group", "layer"), default="matrix_group")
    parser.add_argument("--weight-store", choices=("full_pinned", "pinned_staging"), default="pinned_staging")
    parser.add_argument("--embedding-placement", choices=("resident", "streamed"), default="streamed")
    parser.add_argument("--lm-head-placement", choices=("resident", "streamed"), default="streamed")
    parser.add_argument("--slots", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--gpu-resident-weight-budget", type=parse_byte_size, default=0)
    parser.add_argument("--reference-gpu-resident-weight-budget", type=parse_byte_size)
    parser.add_argument("--candidate-gpu-resident-weight-budget", type=parse_byte_size)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=4e-2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def runtime_args(args, checkpoint, backend, side):
    resident_budget = getattr(
        args, "{}_gpu_resident_weight_budget".format(side)
    )
    if resident_budget is None:
        resident_budget = args.gpu_resident_weight_budget
    return Namespace(
        checkpoint=checkpoint,
        weight_format="auto",
        backend=backend,
        prefill_backend=getattr(args, "{}_prefill_backend".format(side)),
        decode_backend=getattr(args, "{}_decode_backend".format(side)),
        granularity=args.granularity,
        weight_store=args.weight_store,
        embedding_mode=args.embedding_placement,
        lm_head_mode=args.lm_head_placement,
        slots=args.slots,
        block_size=16,
        top_k=args.top_k,
        profile=False,
        gpu_resident_weight_budget=resident_budget,
    )


def geometry_signature(config):
    value = adapter_for_config(config).build_geometry(config).as_dict()
    value.pop("model_id", None)
    return value


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.provider == "cutlass":
        from layer_streaming.providers.cutlass import load_cutlass_w8a16_provider

        load_cutlass_w8a16_provider(args.provider_library)
    reference_config = AutoConfig.from_pretrained(
        args.reference_checkpoint, local_files_only=True
    )
    candidate_config = AutoConfig.from_pretrained(
        args.candidate_checkpoint, local_files_only=True
    )
    if geometry_signature(reference_config) != geometry_signature(candidate_config):
        raise SystemExit("reference and candidate model geometries differ")
    device = torch.device(args.device)
    prefill = torch.tensor([args.input_ids], dtype=torch.long, device=device)
    decode = [
        torch.tensor([[token]], dtype=torch.long, device=device)
        for token in args.decode_ids
    ]
    reference = streaming_outputs(
        runtime_args(
            args, args.reference_checkpoint, args.reference_backend, "reference"
        ),
        reference_config,
        device,
        prefill,
        decode,
    )
    candidate = streaming_outputs(
        runtime_args(
            args, args.candidate_checkpoint, args.candidate_backend, "candidate"
        ),
        candidate_config,
        device,
        prefill,
        decode,
    )
    ok, comparisons = compare(
        reference, candidate, args.atol, args.rtol, args.top_k
    )
    logits = {
        name: value
        for name, value in comparisons.items()
        if name.endswith("/logits")
    }
    report = {
        "schema_version": 1,
        "ok": ok,
        "reference": {
            "checkpoint": str(args.reference_checkpoint.resolve()),
            "backend": args.reference_backend,
            "prefill_backend": (
                args.reference_prefill_backend or args.reference_backend
            ),
            "decode_backend": (
                args.reference_decode_backend or args.reference_backend
            ),
            "gpu_resident_weight_budget_bytes": (
                args.reference_gpu_resident_weight_budget
                if args.reference_gpu_resident_weight_budget is not None
                else args.gpu_resident_weight_budget
            ),
        },
        "candidate": {
            "checkpoint": str(args.candidate_checkpoint.resolve()),
            "backend": args.candidate_backend,
            "prefill_backend": (
                args.candidate_prefill_backend or args.candidate_backend
            ),
            "decode_backend": (
                args.candidate_decode_backend or args.candidate_backend
            ),
            "gpu_resident_weight_budget_bytes": (
                args.candidate_gpu_resident_weight_budget
                if args.candidate_gpu_resident_weight_budget is not None
                else args.gpu_resident_weight_budget
            ),
        },
        "provider": args.provider,
        "input_ids": args.input_ids,
        "decode_ids": args.decode_ids,
        "atol": args.atol,
        "rtol": args.rtol,
        "summary": {
            "comparison_count": len(comparisons),
            "failed_count": sum(
                not item.get("ok", False) for item in comparisons.values()
            ),
            "logits": logits,
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "comparisons": "written to output"}, indent=2))
    if not ok and not args.allow_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
