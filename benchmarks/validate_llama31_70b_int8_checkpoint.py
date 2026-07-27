#!/usr/bin/env python3
"""Validate the 70B W8A8 checkpoint against Cascade's mixed-dtype plan."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import build_llama31_70b_int8_plan  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = build_llama31_70b_int8_plan("matrix")
    index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    weight_map = index["weight_map"]
    actual = {}
    files = sorted(set(weight_map.values()))
    for filename in files:
        keys = [
            key for key, value in weight_map.items() if value == filename
        ]
        with safe_open(
            str(args.checkpoint / filename),
            framework="pt",
            device="cpu",
        ) as source:
            for key in keys:
                tensor = source.get_slice(key)
                actual[key] = {
                    "shape": tuple(tensor.get_shape()),
                    "dtype": tensor.get_dtype(),
                }

    expected_keys = set(plan.tensors)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    mismatches = []
    dtype_names = {"int8": "I8", "bfloat16": "BF16"}
    for key in sorted(expected_keys & actual_keys):
        spec = plan.tensors[key]
        metadata = actual[key]
        if (
            tuple(spec.shape) != metadata["shape"]
            or dtype_names[spec.dtype] != metadata["dtype"]
        ):
            mismatches.append(
                {
                    "key": key,
                    "expected_shape": list(spec.shape),
                    "actual_shape": list(metadata["shape"]),
                    "expected_dtype": dtype_names[spec.dtype],
                    "actual_dtype": metadata["dtype"],
                }
            )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "safetensor_shards": len(files),
        "expected_tensor_count": len(expected_keys),
        "actual_tensor_count": len(actual_keys),
        "dtype_counts": dict(
            Counter(item["dtype"] for item in actual.values())
        ),
        "tensor_payload_bytes": sum(
            spec.nbytes for spec in plan.tensors.values()
        ),
        "plan_host_arena_bytes": plan.host_arena_bytes,
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
        "status": (
            "passed"
            if not missing and not extra and not mismatches
            else "failed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
