#!/usr/bin/env python3
"""Aggregate RGKV quality evidence without executing a model or CUDA work."""

import argparse
from collections import Counter
import json
import math
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


SCHEMA_VERSION = 1
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT_RUN"
PARTIAL = "PARTIAL"
PLAN_ONLY = "PLAN_ONLY"

QUALITY_LEVELS = (
    "selection",
    "attention",
    "logit",
    "ppl",
    "long_context",
)
BUDGET_RATIOS = (0.75, 0.50, 0.25, 0.125)
RECENT_PAGES = (2, 4, 8, 16)

# These are versioned initial gates, not hidden implementation constants.  Every
# threshold and comparison is repeated in cases.json and report.md.
GATES = {
    "selection": {
        "ranking_agreement": (">=", 0.95),
        "topk_overlap": (">=", 0.90),
        "selected_page_agreement": (">=", 0.90),
    },
    "attention": {
        "max_abs_error": ("<=", 0.05),
        "mean_abs_error": ("<=", 0.01),
        "cosine_similarity": (">=", 0.99),
    },
    "logit": {
        "top1_match_rate": (">=", 0.95),
        "topk_overlap": (">=", 0.90),
        "logit_cosine": (">=", 0.99),
        "kl_divergence": ("<=", 0.05),
    },
    "ppl": {
        "dense_perplexity": (">", 0.0),
        "rgkv_perplexity": (">", 0.0),
        "relative_ppl_degradation": ("<=", 0.05),
    },
    "long_context": {
        "needle_recall_early": (">=", 0.90),
        "needle_recall_middle": (">=", 0.90),
        "needle_recall_late": (">=", 0.90),
        "retrieval_accuracy": (">=", 0.90),
        "long_qa_score": (">=", 0.90),
        "multi_turn_score": (">=", 0.90),
    },
}

METRIC_RANGES = {
    "ranking_agreement": (0.0, 1.0),
    "topk_overlap": (0.0, 1.0),
    "selected_page_agreement": (0.0, 1.0),
    "max_abs_error": (0.0, None),
    "mean_abs_error": (0.0, None),
    "cosine_similarity": (-1.0, 1.0),
    "top1_match_rate": (0.0, 1.0),
    "logit_cosine": (-1.0, 1.0),
    "kl_divergence": (0.0, None),
    "dense_perplexity": (0.0, None),
    "rgkv_perplexity": (0.0, None),
    "relative_ppl_degradation": (None, None),
    "needle_recall_early": (0.0, 1.0),
    "needle_recall_middle": (0.0, 1.0),
    "needle_recall_late": (0.0, 1.0),
    "retrieval_accuracy": (0.0, 1.0),
    "long_qa_score": (0.0, 1.0),
    "multi_turn_score": (0.0, 1.0),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan", "aggregate"), default="plan"
    )
    parser.add_argument(
        "--target", choices=("offline", "real-70b"), default="offline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="JSON evidence document; required by aggregate mode",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="plan metadata only; this runner never opens model weights",
    )
    parser.add_argument(
        "--missing-status",
        choices=(NOT_RUN, BLOCKED),
        default=NOT_RUN,
        help="status assigned to every absent matrix cell",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "rgkv_quality",
    )
    return parser


def validate_args(args):
    if args.mode == "aggregate" and args.input is None:
        raise ValueError("aggregate mode requires --input")
    if args.mode == "plan" and args.input is not None:
        raise ValueError("plan mode does not consume --input")
    if args.target == "real-70b" and args.mode != "plan":
        raise ValueError(
            "real-70b is plan-only in this runner; execute it with the "
            "exclusive-GPU model qualification harness"
        )
    return args


def planned_cases():
    return tuple(
        {
            "case_id": "rgkv-{}-b{:04d}-recent{}".format(
                level, int(round(ratio * 1000.0)), recent
            ),
            "level": level,
            "budget_ratio": ratio,
            "budget_percent": ratio * 100.0,
            "recent_pages": recent,
        }
        for ratio in BUDGET_RATIOS
        for recent in RECENT_PAGES
        for level in QUALITY_LEVELS
    )


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _compare(value, operator, threshold):
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    if operator == ">":
        return value > threshold
    raise ValueError("unsupported gate operator: {}".format(operator))


def _in_metric_range(name, value):
    lower, upper = METRIC_RANGES[name]
    return (
        (lower is None or float(value) >= lower)
        and (upper is None or float(value) <= upper)
    )


def _normalize_metrics(level, raw_metrics):
    metrics = dict(raw_metrics or {})
    if level == "ppl":
        dense = metrics.get("dense_perplexity")
        candidate = metrics.get("rgkv_perplexity")
        if _finite_number(dense) and float(dense) > 0 and _finite_number(candidate):
            computed = (float(candidate) - float(dense)) / float(dense)
            supplied = metrics.get("relative_ppl_degradation")
            if supplied is not None and (
                not _finite_number(supplied)
                or not math.isclose(
                    float(supplied), computed, rel_tol=1e-6, abs_tol=1e-9
                )
            ):
                metrics["relative_ppl_degradation_mismatch"] = True
            metrics["relative_ppl_degradation"] = computed
    return metrics


def evaluate_evidence(level, raw_metrics):
    """Return a strict gate result; absent or malformed metrics never pass."""

    metrics = _normalize_metrics(level, raw_metrics)
    evaluations = []
    missing = []
    invalid = []
    for name, (operator, threshold) in GATES[level].items():
        value = metrics.get(name)
        if value is None:
            missing.append(name)
            continue
        if not _finite_number(value) or not _in_metric_range(name, value):
            invalid.append(name)
            continue
        evaluations.append(
            {
                "metric": name,
                "value": float(value),
                "operator": operator,
                "threshold": threshold,
                "passed": _compare(float(value), operator, threshold),
            }
        )
    mismatch = bool(metrics.get("relative_ppl_degradation_mismatch", False))
    passed = (
        not missing
        and not invalid
        and not mismatch
        and len(evaluations) == len(GATES[level])
        and all(item["passed"] for item in evaluations)
    )
    return {
        "status": PASS if passed else FAIL,
        "metrics": metrics,
        "gate": {
            "schema": GATES[level],
            "evaluations": evaluations,
            "missing_metrics": missing,
            "invalid_metrics": invalid,
            "derived_metric_mismatch": mismatch,
            "passed": passed,
        },
    }


def _canonical_ratio(value):
    value = float(value)
    for candidate in BUDGET_RATIOS:
        if math.isclose(value, candidate, rel_tol=0.0, abs_tol=1e-9):
            return candidate
    raise ValueError("unsupported budget_ratio: {}".format(value))


def _load_records(document):
    records = document.get("results") if isinstance(document, dict) else document
    if not isinstance(records, list):
        raise ValueError("input must be a list or an object containing results[]")
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every result must be an object")
        level = str(record.get("level", ""))
        if level not in QUALITY_LEVELS:
            raise ValueError("unsupported quality level: {}".format(level))
        ratio = _canonical_ratio(record.get("budget_ratio"))
        recent = int(record.get("recent_pages"))
        if recent not in RECENT_PAGES:
            raise ValueError("unsupported recent_pages: {}".format(recent))
        key = (level, ratio, recent)
        if key in indexed:
            raise ValueError("duplicate evidence for {}".format(key))
        indexed[key] = record
    return indexed


def aggregate_cases(document, missing_status=NOT_RUN, missing_reason=None):
    indexed = _load_records(document)
    cases = []
    for plan in planned_cases():
        key = (plan["level"], plan["budget_ratio"], plan["recent_pages"])
        record = indexed.get(key)
        if record is None:
            cases.append(
                dict(
                    plan,
                    status=missing_status,
                    reason=missing_reason or "no evidence was provided",
                    metrics={},
                    gate={
                        "schema": GATES[plan["level"]],
                        "evaluations": [],
                        "missing_metrics": list(GATES[plan["level"]]),
                        "invalid_metrics": [],
                        "derived_metric_mismatch": False,
                        "passed": False,
                    },
                    evidence_class="NONE",
                )
            )
            continue
        supplied_status = record.get("status")
        if supplied_status not in (None, PASS, FAIL, BLOCKED, NOT_RUN):
            raise ValueError(
                "unsupported evidence status: {}".format(supplied_status)
            )
        if supplied_status in (BLOCKED, NOT_RUN):
            cases.append(
                dict(
                    plan,
                    status=supplied_status,
                    reason=record.get("reason") or "evidence was not executed",
                    metrics=dict(record.get("metrics") or {}),
                    gate={
                        "schema": GATES[plan["level"]],
                        "evaluations": [],
                        "missing_metrics": list(GATES[plan["level"]]),
                        "invalid_metrics": [],
                        "derived_metric_mismatch": False,
                        "passed": False,
                    },
                    evidence_class=record.get("evidence_class", "NONE"),
                )
            )
            continue
        evaluated = evaluate_evidence(plan["level"], record.get("metrics"))
        status = evaluated["status"]
        if supplied_status == FAIL:
            status = FAIL
        cases.append(
            dict(
                plan,
                status=status,
                reason=(
                    record.get("reason")
                    if status == FAIL and supplied_status == FAIL
                    else None if status == PASS else "quality thresholds were not met"
                ),
                metrics=evaluated["metrics"],
                gate=evaluated["gate"],
                evidence_class=record.get("evidence_class", "MEASURED_INPUT"),
                supplied_status=supplied_status,
            )
        )
    return cases


def build_summary(args, cases):
    counts = Counter(case["status"] for case in cases)
    configurations = []
    for ratio in BUDGET_RATIOS:
        for recent in RECENT_PAGES:
            selected = [
                case for case in cases
                if case["budget_ratio"] == ratio
                and case["recent_pages"] == recent
            ]
            statuses = {case["level"]: case["status"] for case in selected}
            configurations.append(
                {
                    "budget_ratio": ratio,
                    "recent_pages": recent,
                    "level_statuses": statuses,
                    "quality_gate_passed": all(
                        statuses.get(level) == PASS for level in QUALITY_LEVELS
                    ),
                }
            )
    supplied = len(cases) - counts[NOT_RUN] - counts[BLOCKED]
    complete = counts[NOT_RUN] == 0 and counts[BLOCKED] == 0
    qualified = [item for item in configurations if item["quality_gate_passed"]]
    if supplied == 0:
        overall = BLOCKED if counts[BLOCKED] else NOT_RUN
    elif not complete:
        overall = PARTIAL
    elif qualified:
        overall = PASS
    else:
        overall = FAIL
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "mode": args.mode,
        "target": args.target,
        "status": overall,
        "evidence_class": PLAN_ONLY if args.mode == "plan" else "AGGREGATED",
        "case_count": len(cases),
        "status_counts": {
            status: int(counts[status])
            for status in (PASS, FAIL, BLOCKED, NOT_RUN)
        },
        "budget_ratios": list(BUDGET_RATIOS),
        "recent_pages": list(RECENT_PAGES),
        "quality_levels": list(QUALITY_LEVELS),
        "complete_budget_sweep": complete,
        "qualified_configuration_count": len(qualified),
        "qualified_configurations": [
            {
                "budget_ratio": item["budget_ratio"],
                "recent_pages": item["recent_pages"],
            }
            for item in qualified
        ],
        "production_default_allowed": False,
        "production_default_blocker": (
            "this quality-only aggregator does not prove lifecycle, memory, "
            "exclusive-GPU execution, or positive end-to-end performance"
        ),
        "configurations": configurations,
    }


def render_markdown(summary, cases):
    lines = [
        "# RGKV Quality Gate",
        "",
        "Status: `{}`  ".format(summary["status"]),
        "Evidence class: `{}`  ".format(summary["evidence_class"]),
        "Production default allowed: `false`",
        "",
        "This report aggregates supplied measurements; it does not execute a model, CUDA workload, or 70B qualification case.",
        "",
        "## Budget sweep",
        "",
        "| Budget | Recent | Selection | Attention | Logit | PPL | Long context | Gate |",
        "|---:|---:|---|---|---|---|---|---|",
    ]
    labels = {
        "selection": "Selection",
        "attention": "Attention",
        "logit": "Logit",
        "ppl": "PPL",
        "long_context": "Long context",
    }
    for config in summary["configurations"]:
        status = config["level_statuses"]
        lines.append(
            "| {:.1f}% | {} | {} | {} | {} | {} | {} | {} |".format(
                config["budget_ratio"] * 100.0,
                config["recent_pages"],
                status["selection"],
                status["attention"],
                status["logit"],
                status["ppl"],
                status["long_context"],
                PASS if config["quality_gate_passed"] else "NOT_QUALIFIED",
            )
        )
    lines.extend(["", "## Gate definitions", ""])
    for level in QUALITY_LEVELS:
        lines.append("### {}".format(labels[level]))
        lines.append("")
        for metric, (operator, threshold) in GATES[level].items():
            lines.append("- `{}` {} `{}`".format(metric, operator, threshold))
        lines.append("")
    non_pass = [case for case in cases if case["status"] != PASS]
    lines.extend(
        [
            "## Evidence limitations",
            "",
            "- Missing matrix cells remain explicit `NOT_RUN` or `BLOCKED` entries.",
            "- A supplied `PASS` label cannot override missing, invalid, or below-threshold metrics.",
            "- Long-context evidence covers early/middle/late needle recall, retrieval, long QA, and multi-turn continuation.",
            "- Quality PASS alone cannot enable the production default; lifecycle, memory, Tier compatibility, exclusive-GPU execution, and positive E2E gain remain separate gates.",
            "- Non-PASS case count: `{}`.".format(len(non_pass)),
            "",
        ]
    )
    return "\n".join(lines)


def run(args):
    args = validate_args(args)
    environment = capture_cuda_environment(ROOT, args.device)
    environment.update(
        {
            "runner": "qualify_rgkv_quality.py",
            "execution_performed": False,
            "model_weights_opened": False,
            "checkpoint_configured": bool(args.checkpoint),
            "checkpoint_path_recorded": False,
            "target": args.target,
        }
    )
    if args.mode == "plan":
        missing_status = BLOCKED if args.target == "real-70b" else args.missing_status
        reason = (
            "PLAN_ONLY: real 70B quality execution requires the dedicated "
            "exclusive-GPU model harness"
            if args.target == "real-70b"
            else "PLAN_ONLY: no measurement input was requested"
        )
        cases = aggregate_cases([], missing_status, reason)
    else:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        cases = aggregate_cases(document, args.missing_status)
    summary = build_summary(args, cases)
    markdown = render_markdown(summary, cases)
    paths = write_report_bundle(
        args.output_dir, environment, cases, summary, markdown
    )
    return summary, cases, paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary, _cases, paths = run(args)
    print(json.dumps({"status": summary["status"], "paths": paths}, sort_keys=True))
    return 0 if summary["status"] in {PASS, NOT_RUN, BLOCKED, PARTIAL} else 1


if __name__ == "__main__":
    raise SystemExit(main())
