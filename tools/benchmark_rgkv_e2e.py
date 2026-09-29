#!/usr/bin/env python3
"""Plan or aggregate Dense-vs-RGKV end-to-end decode performance evidence."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qualification_common import utc_now, write_report_bundle  # noqa: E402


CONTEXTS = (2048, 8192, 16384)
DECODES = (32, 128)
BUDGET_RATIOS = (0.75, 0.50, 0.25, 0.125)
RECENT_PAGES = (2, 4, 8, 16)
FIELDS = (
    "dense_attention_ms",
    "rgkv_score_ms",
    "rgkv_topk_ms",
    "rgkv_prefetch_ms",
    "rgkv_attention_ms",
    "dense_decode_ms",
    "rgkv_decode_ms",
    "dense_h2d_kv_bytes",
    "rgkv_h2d_kv_bytes",
    "rgkv_index_bytes",
    "rgkv_cpu_sync_count",
)


def planned_cases():
    return tuple(
        {
            "case_id": "c{}_d{}_b{:04d}_r{}".format(
                context, decode, int(ratio * 1000), recent
            ),
            "context_tokens": context,
            "decode_tokens": decode,
            "budget_ratio": ratio,
            "recent_pages": recent,
        }
        for context in CONTEXTS
        for decode in DECODES
        for ratio in BUDGET_RATIOS
        for recent in RECENT_PAGES
    )


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate(record):
    metrics = dict(record.get("metrics") or {})
    missing = [name for name in FIELDS if not _finite(metrics.get(name))]
    if missing:
        return "FAIL", missing, metrics
    dense = float(metrics["dense_decode_ms"])
    rgkv = float(metrics["rgkv_decode_ms"])
    if dense <= 0 or rgkv <= 0:
        return "FAIL", ["positive_decode_latency"], metrics
    components = sum(
        float(metrics[name])
        for name in (
            "rgkv_score_ms",
            "rgkv_topk_ms",
            "rgkv_prefetch_ms",
            "rgkv_attention_ms",
        )
    )
    metrics["rgkv_accounted_component_ms"] = components
    metrics["e2e_gain_ms"] = dense - rgkv
    metrics["e2e_speedup"] = dense / rgkv
    metrics["kv_h2d_saved_bytes"] = (
        float(metrics["dense_h2d_kv_bytes"])
        - float(metrics["rgkv_h2d_kv_bytes"])
    )
    passed = (
        metrics["e2e_gain_ms"] > 0
        and int(metrics["rgkv_cpu_sync_count"]) == 0
        and components <= rgkv * 1.05
        and metrics["kv_h2d_saved_bytes"] >= 0
    )
    return ("PASS" if passed else "FAIL"), [], metrics


def aggregate(document):
    records = document.get("results", document) if isinstance(document, dict) else document
    if not isinstance(records, list):
        raise ValueError("performance input must contain results[]")
    by_id = {}
    for record in records:
        case_id = str(record.get("case_id"))
        if case_id in by_id:
            raise ValueError("duplicate performance case {}".format(case_id))
        by_id[case_id] = record
    result = []
    for plan in planned_cases():
        record = by_id.pop(plan["case_id"], None)
        if record is None:
            result.append(dict(plan, status="NOT_RUN", metrics={}, reason="no evidence"))
            continue
        status, missing, metrics = evaluate(record)
        result.append(
            dict(
                plan,
                status=status,
                metrics=metrics,
                reason=(None if status == "PASS" else "missing/failed performance gate"),
                missing_metrics=missing,
            )
        )
    if by_id:
        raise ValueError("unknown performance cases: {}".format(sorted(by_id)))
    return result


def summary(cases, mode):
    passed = [case for case in cases if case["status"] == "PASS"]
    complete = all(case["status"] != "NOT_RUN" for case in cases)
    best = None
    if passed:
        winner = min(passed, key=lambda case: case["metrics"]["rgkv_decode_ms"])
        best = {
            name: winner[name]
            for name in ("case_id", "context_tokens", "decode_tokens", "budget_ratio", "recent_pages")
        }
        best.update(
            {
                "rgkv_decode_ms": winner["metrics"]["rgkv_decode_ms"],
                "e2e_speedup": winner["metrics"]["e2e_speedup"],
            }
        )
    gains = [case["metrics"]["e2e_gain_ms"] for case in passed]
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "mode": mode,
        "status": "PASS" if complete and passed else "PARTIAL" if passed else "NOT_RUN",
        "case_count": len(cases),
        "pass_count": len(passed),
        "complete": complete,
        "best_measured_configuration": best,
        "mean_positive_e2e_gain_ms": statistics.mean(gains) if gains else None,
        "production_default_allowed": False,
        "production_default_blocker": "quality, lifecycle, stability and exclusive 70B gates remain separate",
    }


def render(result, cases):
    lines = [
        "# RGKV End-to-End Performance Gate", "",
        "Status: `{}`".format(result["status"]), "",
        "Every PASS includes selection, top-k, selected-only prefetch, attention and total decode latency.", "",
        "| Case | Context | Decode | Budget | Recent | Status | Dense ms | RGKV ms | Speedup | KV H2D saved |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        metrics = case.get("metrics") or {}
        lines.append(
            "| {} | {} | {} | {:.1f}% | {} | {} | {} | {} | {} | {} |".format(
                case["case_id"], case["context_tokens"], case["decode_tokens"],
                case["budget_ratio"] * 100, case["recent_pages"], case["status"],
                metrics.get("dense_decode_ms", "-"), metrics.get("rgkv_decode_ms", "-"),
                metrics.get("e2e_speedup", "-"), metrics.get("kv_h2d_saved_bytes", "-"),
            )
        )
    lines.extend(["", "Production default allowed: `false`", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "aggregate"), default="plan")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/rgkv_e2e")
    args = parser.parse_args(argv)
    if args.mode == "aggregate" and args.input is None:
        parser.error("aggregate requires --input")
    cases = (
        [dict(case, status="NOT_RUN", metrics={}, reason="PLAN_ONLY") for case in planned_cases()]
        if args.mode == "plan"
        else aggregate(json.loads(args.input.read_text(encoding="utf-8")))
    )
    result = summary(cases, args.mode)
    environment = {
        "schema_version": 1,
        "generated_at": result["generated_at"],
        "execution_performed": False,
        "evidence_note": "aggregate reads measurements; plan executes no model",
    }
    paths = write_report_bundle(args.output_dir, environment, cases, result, render(result, cases))
    print(json.dumps({"summary": result, "paths": paths}, sort_keys=True))
    return 0 if result["status"] in {"NOT_RUN", "PARTIAL", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
