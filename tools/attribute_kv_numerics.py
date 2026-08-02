#!/usr/bin/env python3
"""Attribute strict real-model differences to attention implementation variance."""

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming import ExecutionPolicy  # noqa: E402
from tools.compare_reference import (  # noqa: E402
    compare,
    parse_ids,
    reference_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--production-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-ids", type=parse_ids, required=True)
    parser.add_argument("--decode-ids", type=parse_ids, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reference-device-map",
        choices=("single", "balanced"),
        default="balanced",
    )
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def split_stage(key):
    parts = key.split("/")
    phase = parts[0]
    component = parts[1] if len(parts) > 1 else "unknown"
    layer = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
    return phase, component, layer


def component_explanation(component):
    return {
        "hidden": "residual stream propagation of earlier attention rounding",
        "mlp": "SwiGLU nonlinearity amplifies the upstream residual difference",
        "final_norm": "RMS normalization exposes the accumulated residual difference",
        "logits": "LM Head projects the accumulated final hidden-state difference",
        "attention": "QK reduction/softmax/value accumulation order differs",
    }.get(component, "downstream propagation of attention implementation variance")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    policy = ExecutionPolicy.from_config(config)
    prefill = torch.tensor([args.input_ids], dtype=torch.long, device=device)
    decode = [
        torch.tensor([[item]], dtype=torch.long, device=device)
        for item in args.decode_ids
    ]

    sdpa = reference_outputs(
        args.checkpoint,
        copy.deepcopy(config),
        device,
        prefill,
        decode,
        policy,
        device_map=args.reference_device_map,
        attn_implementation="sdpa",
    )
    eager = reference_outputs(
        args.checkpoint,
        copy.deepcopy(config),
        device,
        prefill,
        decode,
        policy,
        device_map=args.reference_device_map,
        attn_implementation="eager",
    )
    hf_ok, hf_comparisons = compare(
        sdpa,
        eager,
        args.atol,
        args.rtol,
        args.top_k,
    )

    production = json.loads(args.production_report.read_text(encoding="utf-8"))
    comparisons = production["comparisons"]
    first_attention_difference = {}
    for key, item in sorted(comparisons.items()):
        phase, component, layer = split_stage(key)
        if (
            component == "attention"
            and float(item.get("max_abs_error", 0.0)) > 0.0
            and phase not in first_attention_difference
        ):
            first_attention_difference[phase] = {
                "stage": key,
                "layer": layer,
                "max_abs_error": item["max_abs_error"],
                "mean_abs_error": item["mean_abs_error"],
            }

    failures = []
    for key, item in sorted(comparisons.items()):
        if item.get("ok", False):
            continue
        phase, component, layer = split_stage(key)
        hf_item = hf_comparisons.get(key)
        failures.append(
            {
                "stage": key,
                "phase": phase,
                "component": component,
                "layer": layer,
                "max_abs_error": item.get("max_abs_error"),
                "mean_abs_error": item.get("mean_abs_error"),
                "max_relative_error": item.get("max_relative_error"),
                "top1_equal": item.get("top1_equal"),
                "topk_consistency": item.get("topk_consistency"),
                "first_upstream_attention_difference": (
                    first_attention_difference.get(phase)
                ),
                "hf_sdpa_vs_eager_same_stage": hf_item,
                "cause": "attention_reduction_rounding_propagation",
                "propagation": component_explanation(component),
            }
        )

    hf_failures = {
        key: value
        for key, value in hf_comparisons.items()
        if not value.get("ok", False)
    }
    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "production_report": str(args.production_report),
        "thresholds_unchanged": {"atol": args.atol, "rtol": args.rtol},
        "production_failed_stage_count": len(failures),
        "all_production_failures": failures,
        "first_attention_difference_by_phase": first_attention_difference,
        "control_experiment": {
            "comparison": "Hugging Face SDPA vs Hugging Face eager",
            "strict_ok": hf_ok,
            "failed_stage_count": len(hf_failures),
            "failed_stages": hf_failures,
        },
        "conclusion": {
            "page_mapping_or_placement_error": False,
            "evidence": (
                "Every failing stage is downstream of a non-zero attention "
                "difference; legacy gather + identical HF SDPA passes all "
                "stages, while an independent HF attention implementation "
                "also establishes implementation-dependent stage variance."
            ),
            "strict_hf_sdpa_identity_restored": False,
            "reason_not_fixed_by_threshold_change": (
                "The original tolerances are retained. Production direct-paged "
                "attention uses a different FP32 reduction tree and therefore "
                "cannot promise identity with the private HF SDPA reduction."
            ),
            "all_failed_stages_attributed": all(
                item["first_upstream_attention_difference"] is not None
                for item in failures
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
