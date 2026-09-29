#!/usr/bin/env python3
"""Write blocked Dense/Tiered qualification plans without executing CUDA cases."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qualification_common import (  # noqa: E402
    capture_cuda_environment,
    utc_now,
    write_report_bundle,
)


def _cases(kind, blocker_detail=None):
    identifiers = {
        "dense": (
            ("DENSE-COW-CROSS-STREAM", "COW cross-stream release ordering"),
            ("DENSE-ATTENTION-FENCE", "attention immediate reset/release"),
            ("DENSE-APPEND-LAYER-FAILURE", "Append layer 1/20/40/79 failure"),
            ("DENSE-ERROR-MATRIX", "query/sync/cancel/timeout/double drain"),
        ),
        "tiered": (
            ("TIERED-SMALL-HOT-A-B", "GPU-only Dense vs Tiered exact A/B"),
            ("TIERED-H2D-D2H-FAULTS", "cancel/timeout/epoch/allocation faults"),
            ("TIERED-SESSION-STABILITY", "session/reset/close/continuation"),
            ("TIERED-70B-LONG-CONTEXT", "real 70B 8K/16K Tiered context"),
        ),
    }[kind]
    reason = blocker_detail or (
        "BLOCKED_NO_RESERVATION: point-in-time idle GPU snapshot is not an "
        "exclusive scheduler/server reservation"
    )
    return [
        {
            "case_id": case_id,
            "name": name,
            "status": "BLOCKED",
            "evidence": "PLAN_ONLY",
            "reason": reason,
            "metrics": {},
        }
        for case_id, name in identifiers
    ]


def _markdown(kind, summary, cases, blocker_detail=None):
    title = "Dense GPU" if kind == "dense" else "Tiered KV"
    lines = [
        "# {} Exclusive Qualification Plan".format(title),
        "",
        "Status: `BLOCKED_NO_RESERVATION`  ",
        "Evidence class: `PLAN_ONLY`",
        "",
        "No CUDA qualification case was executed. An idle `nvidia-smi` snapshot does not prove exclusive ownership.",
        "",
    ]
    if blocker_detail:
        lines.extend(
            [
                "Scheduler admission evidence: `{}`".format(
                    blocker_detail.replace("`", "'")
                ),
                "",
            ]
        )
    lines.extend([
        "| Case | Status | Reason |",
        "|---|---|---|",
    ])
    for case in cases:
        lines.append(
            "| {} | `{}` | {} |".format(
                case["case_id"], case["status"], case["reason"]
            )
        )
    lines.extend(
        [
            "",
            "Run the qualification harness only after recording a scheduler/server reservation ID and start/end external-process evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dense-output-dir",
        default=str(ROOT / "reports" / "kv_cuda_qualification"),
    )
    parser.add_argument(
        "--tiered-output-dir",
        default=str(ROOT / "reports" / "kv_tiered_qualification"),
    )
    parser.add_argument(
        "--blocker-detail",
        help="audited scheduler/server admission failure; no command is run",
    )
    args = parser.parse_args(argv)
    environment = capture_cuda_environment(ROOT, args.device)
    environment["reservation_evidence_present"] = False
    environment["qualification_execution_authorized"] = False
    environment["qualification_blocker"] = args.blocker_detail
    for kind, output in (
        ("dense", args.dense_output_dir),
        ("tiered", args.tiered_output_dir),
    ):
        cases = _cases(kind, args.blocker_detail)
        summary = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "mode": "plan",
            "status": "BLOCKED_NO_RESERVATION",
            "evidence_class": "PLAN_ONLY",
            "case_count": len(cases),
            "pass_count": 0,
            "blocked_count": len(cases),
            "qualification_admitted": False,
            "qualification_executed": False,
        }
        write_report_bundle(
            output,
            environment,
            cases,
            summary,
            _markdown(kind, summary, cases, args.blocker_detail),
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
