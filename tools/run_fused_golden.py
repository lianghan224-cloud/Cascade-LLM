#!/usr/bin/env python3
"""Run every registered real-checkpoint fused W8A16 golden case."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming.numerical_contract import (  # noqa: E402
    evaluate_fused_diagnostic,
)


DEFAULT_CONTRACT = (
    PROJECT_ROOT / "tests/fixtures/fused_w8a16_sm86_golden_v1.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--provider-library")
    parser.add_argument("--gpu-resident-weight-budget", default="8GiB")
    parser.add_argument("--slots", type=int, default=2)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in contract["golden_cases"]:
        output = args.output_dir / (case["name"] + ".json")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools/diagnose_fused_reference.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--input-ids",
            ",".join(str(item) for item in case["input_ids"]),
            "--decode-ids",
            ",".join(str(item) for item in case["decode_ids"]),
            "--device",
            args.device,
            "--granularity",
            "matrix_group",
            "--weight-store",
            "pinned_staging",
            "--slots",
            str(args.slots),
            "--gpu-resident-weight-budget",
            args.gpu_resident_weight_budget,
            "--output",
            str(output),
        ]
        if args.provider_library:
            command.extend(["--provider-library", args.provider_library])
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise SystemExit(
                "golden case {} diagnostic failed with exit {}".format(
                    case["name"], completed.returncode
                )
            )
        diagnostic = json.loads(output.read_text(encoding="utf-8"))
        gate = evaluate_fused_diagnostic(diagnostic, contract)
        gate_output = args.output_dir / (case["name"] + "_gate.json")
        gate_output.write_text(
            json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        cases.append(
            {
                "name": case["name"],
                "diagnostic": str(output),
                "gate": str(gate_output),
                "passed": gate["passed"],
                "violations": gate["violations"],
                "observations": gate["observations"],
            }
        )
    result = {
        "schema_version": 1,
        "contract": contract["name"],
        "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device,
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
    suite_output = args.output_dir / "golden_suite.json"
    suite_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(suite_output),
        "contract": result["contract"],
        "passed": result["passed"],
        "cases": [
            {"name": case["name"], "passed": case["passed"]}
            for case in cases
        ],
    }, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
