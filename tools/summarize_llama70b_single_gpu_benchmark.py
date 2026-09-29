#!/usr/bin/env python3
"""Summarize real single-GPU 70B runs without promoting smoke to qualification."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("baseline", "pipeline", "budget", "stability")


def _percentile(values, fraction):
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _output_hash(token_ids):
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prefill_span(report):
    forwards = (
        report.get("pipeline", {})
        .get("copy_compute_timeline", {})
        .get("per_forward", [])
    )
    for item in forwards:
        if item.get("phase") == "prefill":
            return (item.get("summary") or {}).get("gpu_timeline_span_ms")
    return None


def _admission_valid(case):
    admission = case.get("gpu_admission") or {}
    before = admission.get("before") or {}
    after = admission.get("after") or {}
    boundary_present = bool(before and after)
    boundary_clean = boundary_present and not int(
        before.get("selected_gpu_external_compute_processes") or 0
    ) and not int(after.get("selected_gpu_external_compute_processes") or 0)
    monitor = case.get("gpu_activity_monitor")
    if monitor is None:
        monitor_clean = None
    else:
        monitor_clean = not bool(monitor.get("external_activity_observed"))
    valid = boundary_clean and monitor_clean is not False
    evidence = (
        "CONTINUOUSLY_MONITORED_UNCONTENDED"
        if valid and monitor_clean is True
        else "BOUNDARY_SNAPSHOT_UNCONTENDED"
        if valid
        else "INVALIDATED_EXTERNAL_GPU_ACTIVITY"
        if boundary_present
        else "MISSING_GPU_ACTIVITY_EVIDENCE"
    )
    return valid, evidence


def _row(case, config, source, stage=None):
    valid, activity_evidence = _admission_valid(case)
    raw_path = case.get("raw_report")
    report = None
    if raw_path and Path(raw_path).is_file():
        report = _load(raw_path)
    row = {
        "case_id": case["case_id"],
        "stage": stage,
        "source_bundle": str(source),
        "status": case.get("status"),
        "reason": case.get("reason"),
        "valid_for_uncontended_comparison": bool(
            valid and case.get("status") == "PASS" and report is not None
        ),
        "gpu_activity_evidence": activity_evidence,
        "prompt_tokens": int(case["prompt_tokens"]),
        "decode_tokens": int(case["decode_tokens"]),
        "repetition": int(case.get("repetition", 1)),
        "config": {
            name: config.get(name)
            for name in (
                "weight_store",
                "granularity",
                "slots",
                "prefetch_depth",
                "embedding_placement",
                "lm_head_placement",
                "gpu_resident_weight_budget",
            )
        },
        "metrics": None,
    }
    if report is None:
        return row
    timings = report.get("timings") or {}
    pipeline = report.get("pipeline") or {}
    memory = report.get("memory") or {}
    placement = report.get("transformer_placement") or {}
    generation = report.get("generation") or {}
    kv = pipeline.get("kv") or {}
    latencies = list(timings.get("decode_token_latencies_ms") or [])
    h2d_bytes = pipeline.get("h2d_bytes")
    generated = int((report.get("throughput") or {}).get("generated_tokens") or 0)
    if row["config"].get("gpu_resident_weight_budget") is None:
        row["config"]["gpu_resident_weight_budget"] = placement.get(
            "budget_bytes", 0
        )
    row["metrics"] = {
        "ttft_ms": timings.get("time_to_first_token_ms"),
        "prefill_transformer_gpu_span_ms": _prefill_span(report),
        "prefill_attention_ms": kv.get("prefill_attention_ms"),
        "tpot_mean_ms": statistics.mean(latencies) if latencies else None,
        "tpot_p50_ms": _percentile(latencies, 0.50),
        "tpot_p90_ms": _percentile(latencies, 0.90),
        "h2d_bytes": h2d_bytes,
        "h2d_bytes_per_generated_token": (
            float(h2d_bytes) / generated
            if isinstance(h2d_bytes, (int, float)) and generated
            else None
        ),
        "h2d_time_ms": timings.get("h2d_time_ms"),
        "gpu_idle_or_host_overhead_ms": timings.get(
            "gpu_idle_or_host_overhead_ms"
        ),
        "gpu_peak_bytes": memory.get("gpu_peak_memory_bytes"),
        "kv_bytes": memory.get("kv_cache_bytes"),
        "resident_transformer_weight_bytes": placement.get(
            "resident_weight_bytes"
        ),
        "resident_weight_arena_bytes": placement.get("resident_arena_bytes"),
        "output_hash": _output_hash(generation.get("generated_token_ids") or []),
        "backend_fallbacks": list(pipeline.get("fallback_backends") or []),
        "kv_provider_fallback_count": int(
            kv.get("provider_fallback_count") or 0
        ),
        "kv_reference_fallback_count": int(
            kv.get("provider_reference_fallback_count") or 0
        ),
        "load_time_seconds": timings.get("load_time_seconds"),
    }
    return row


def collect(input_root):
    rows = []
    input_root = Path(input_root)
    for cases_path in sorted(input_root.glob("**/cases.json")):
        if cases_path.parent == input_root:
            continue
        summary_path = cases_path.with_name("summary.json")
        if not summary_path.is_file():
            continue
        summary = _load(summary_path)
        config = summary.get("runner_config") or {}
        relative_parts = cases_path.parent.relative_to(input_root).parts
        stage = (
            relative_parts[0]
            if relative_parts and relative_parts[0] in STAGES
            else None
        )
        for case in _load(cases_path).get("cases", []):
            rows.append(_row(case, config, cases_path.parent, stage=stage))
    return rows


def _valid_stage_rows(rows, stage):
    return [
        row
        for row in rows
        if row.get("stage") == stage
        and row.get("valid_for_uncontended_comparison")
        and row.get("gpu_activity_evidence")
        == "CONTINUOUSLY_MONITORED_UNCONTENDED"
        and row.get("metrics") is not None
    ]


def _performance_key(row):
    metrics = row["metrics"]
    return (
        float(metrics.get("tpot_mean_ms") or math.inf),
        float(metrics.get("gpu_idle_or_host_overhead_ms") or math.inf),
        float(metrics.get("h2d_time_ms") or math.inf),
        float(metrics.get("gpu_peak_bytes") or math.inf),
    )


def select_pipeline_configuration(rows):
    candidates = _valid_stage_rows(rows, "pipeline")
    coverage = {
        (
            row["config"].get("granularity"),
            row["config"].get("slots"),
            row["config"].get("prefetch_depth"),
        )
        for row in candidates
    }
    if len(coverage) != 18:
        return None
    winner = min(candidates, key=_performance_key)
    return {
        "granularity": winner["config"]["granularity"],
        "slots": int(winner["config"]["slots"]),
        "prefetch_depth": int(winner["config"]["prefetch_depth"]),
        "selection_metric": "minimum TPOT; tie-break GPU idle, H2D time, GPU peak",
        "source_case": winner["case_id"],
        "tpot_mean_ms": winner["metrics"]["tpot_mean_ms"],
        "gpu_idle_or_host_overhead_ms": winner["metrics"].get(
            "gpu_idle_or_host_overhead_ms"
        ),
        "h2d_time_ms": winner["metrics"].get("h2d_time_ms"),
    }


def select_budget_configuration(rows):
    candidates = _valid_stage_rows(rows, "budget")
    coverage = {
        int(row["config"].get("gpu_resident_weight_budget") or 0)
        for row in candidates
    }
    expected = {value * 1024 ** 3 for value in (0, 4, 8, 12, 16, 20)}
    if coverage != expected:
        return None
    winner = min(
        candidates,
        key=lambda row: _performance_key(row)
        + (int(row["config"].get("gpu_resident_weight_budget") or 0),),
    )
    budget_bytes = int(
        winner["config"].get("gpu_resident_weight_budget") or 0
    )
    return {
        "granularity": winner["config"]["granularity"],
        "slots": int(winner["config"]["slots"]),
        "prefetch_depth": int(winner["config"]["prefetch_depth"]),
        "gpu_resident_weight_budget_bytes": budget_bytes,
        "gpu_resident_weight_budget_gib": budget_bytes / 1024 ** 3,
        "actual_resident_transformer_weight_bytes": winner["metrics"].get(
            "resident_transformer_weight_bytes"
        ),
        "selection_metric": "minimum TPOT; tie-break GPU idle, H2D time, GPU peak, lower budget",
        "source_case": winner["case_id"],
        "tpot_mean_ms": winner["metrics"]["tpot_mean_ms"],
        "h2d_bytes_per_generated_token": winner["metrics"].get(
            "h2d_bytes_per_generated_token"
        ),
        "gpu_peak_bytes": winner["metrics"].get("gpu_peak_bytes"),
    }


def summarize_stability(rows):
    candidates = sorted(
        _valid_stage_rows(rows, "stability"),
        key=lambda row: int(row.get("repetition") or 0),
    )
    repetitions = {int(row.get("repetition") or 0) for row in candidates}
    complete = repetitions == {1, 2, 3, 4, 5}
    hashes = [row["metrics"].get("output_hash") for row in candidates]
    config_signatures = {
        (
            row["config"].get("granularity"),
            int(row["config"].get("slots") or 0),
            int(row["config"].get("prefetch_depth") or 0),
            int(row["config"].get("gpu_resident_weight_budget") or 0),
        )
        for row in candidates
    }
    configuration = None
    if len(config_signatures) == 1:
        granularity, slots, depth, budget = next(iter(config_signatures))
        configuration = {
            "granularity": granularity,
            "slots": slots,
            "prefetch_depth": depth,
            "gpu_resident_weight_budget_bytes": budget,
        }

    def values(name):
        return [
            float(row["metrics"][name])
            for row in candidates
            if row["metrics"].get(name) is not None
        ]

    def distribution(name):
        items = values(name)
        if not items:
            return None
        return {
            "mean": statistics.mean(items),
            "p50": _percentile(items, 0.50),
            "p90": _percentile(items, 0.90),
            "stddev": statistics.pstdev(items),
            "min": min(items),
            "max": max(items),
        }

    def drift(name):
        items = values(name)
        if not items:
            return None
        return {
            "first_to_last_bytes": items[-1] - items[0],
            "range_bytes": max(items) - min(items),
        }

    return {
        "complete": complete,
        "configuration_consistent": bool(complete and configuration is not None),
        "configuration": configuration,
        "valid_repetitions": sorted(repetitions),
        "output_consistent": bool(complete and len(set(hashes)) == 1),
        "output_hashes": hashes,
        "ttft_ms": distribution("ttft_ms"),
        "tpot_mean_ms": distribution("tpot_mean_ms"),
        "gpu_peak_drift": drift("gpu_peak_bytes"),
        "kv_bytes_drift": drift("kv_bytes"),
        "resident_weight_drift": drift("resident_transformer_weight_bytes"),
    }


def build_summary(rows, generated_at):
    pipeline = select_pipeline_configuration(rows)
    budget = select_budget_configuration(rows)
    stability = summarize_stability(rows)
    baseline_coverage = {
        (row["prompt_tokens"], row["decode_tokens"])
        for row in _valid_stage_rows(rows, "baseline")
    }
    required_baseline = {
        (prompt, decode)
        for prompt in (512, 2048, 4096, 8192)
        for decode in (1, 8, 32)
    }
    stability_matches_budget = bool(
        budget is not None
        and stability["configuration"] is not None
        and stability["configuration"]
        == {
            "granularity": budget["granularity"],
            "slots": budget["slots"],
            "prefetch_depth": budget["prefetch_depth"],
            "gpu_resident_weight_budget_bytes": budget[
                "gpu_resident_weight_budget_bytes"
            ],
        }
    )
    groups_complete = {
        "baseline": baseline_coverage == required_baseline,
        "pipeline": pipeline is not None,
        "budget": budget is not None,
        "stability": bool(
            stability["complete"]
            and stability["configuration_consistent"]
            and stability_matches_budget
        ),
    }
    best = None
    reason = "four benchmark groups are not yet complete"
    if all(groups_complete.values()):
        if stability["output_consistent"]:
            best = dict(budget)
            best["stability"] = stability
            reason = (
                "selected by pipeline and budget TPOT, then accepted by five-run output consistency"
            )
        else:
            reason = "five-run stability output hashes are inconsistent"
    return {
        "schema_version": 2,
        "generated_at": generated_at,
        "status": "COMPLETE" if best is not None else "PARTIAL",
        "evidence_class": "UNRESERVED_REAL_MODEL_BENCHMARK",
        "row_count": len(rows),
        "valid_uncontended_count": sum(
            item["valid_for_uncontended_comparison"] for item in rows
        ),
        "invalid_or_failed_count": sum(
            not item["valid_for_uncontended_comparison"] for item in rows
        ),
        "group_completion": groups_complete,
        "pipeline_selection": pipeline,
        "budget_selection": budget,
        "stability": stability,
        "best_configuration": best,
        "best_configuration_reason": reason,
        "qualified": False,
    }


def _ms(value):
    return "-" if value is None else "{:.1f}".format(float(value))


def _gib(value):
    return "-" if value is None else "{:.3f}".format(float(value) / 1024 ** 3)


def render(rows, summary):
    valid = [item for item in rows if item["valid_for_uncontended_comparison"]]
    lines = [
        "# Llama 3.3 70B Single-GPU Benchmark",
        "",
        "- Generated: `{}`".format(summary["generated_at"]),
        "- Evidence: real checkpoint, single request, unreserved diagnostic benchmark",
        "- Valid uncontended rows: `{}` / `{}`".format(len(valid), len(rows)),
        "- Production/qualification claim: `false`",
        "",
        "TPOT is unavailable for Decode=1. Prefill time is the transformer GPU timeline span; TTFT additionally includes first-token finish/LM-head and host overhead.",
        "",
        "| Stage / Case | Config g/s/p/budget GiB | Validity | TTFT ms | Prefill ms | TPOT mean ms | H2D total / token GiB / ms | GPU idle ms | GPU peak GiB | KV GiB | Weight cache / arena GiB | Output hash | Fallback |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        config = row["config"]
        fallback = "backend={} kv={}/ref={}".format(
            ",".join(metrics.get("backend_fallbacks") or []) or "none",
            metrics.get("kv_provider_fallback_count", "-"),
            metrics.get("kv_reference_fallback_count", "-"),
        )
        lines.append(
            "| {} / {} | {}/{}/{}/{} | {} | {} | {} | {} | {} / {} / {} | {} | {} | {} | {} / {} | {} | {} |".format(
                row.get("stage") or "legacy",
                row["case_id"],
                config.get("granularity"),
                config.get("slots"),
                config.get("prefetch_depth"),
                _gib(config.get("gpu_resident_weight_budget")),
                row["gpu_activity_evidence"],
                _ms(metrics.get("ttft_ms")),
                _ms(metrics.get("prefill_transformer_gpu_span_ms")),
                _ms(metrics.get("tpot_mean_ms")),
                _gib(metrics.get("h2d_bytes")),
                _gib(metrics.get("h2d_bytes_per_generated_token")),
                _ms(metrics.get("h2d_time_ms")),
                _ms(metrics.get("gpu_idle_or_host_overhead_ms")),
                _gib(metrics.get("gpu_peak_bytes")),
                _gib(metrics.get("kv_bytes")),
                _gib(metrics.get("resident_transformer_weight_bytes")),
                _gib(metrics.get("resident_weight_arena_bytes")),
                (metrics.get("output_hash") or "-")[:12],
                fallback,
            )
        )
    lines.extend(
        [
            "",
            "Rows marked `INVALIDATED_EXTERNAL_GPU_ACTIVITY` are retained as failure evidence and excluded from configuration selection.",
            "",
            "## Automatic selection",
            "",
            "- Group completion: `{}`".format(
                json.dumps(summary["group_completion"], sort_keys=True)
            ),
            "- Pipeline selection: `{}`".format(
                json.dumps(summary["pipeline_selection"], sort_keys=True)
            ),
            "- Budget selection: `{}`".format(
                json.dumps(summary["budget_selection"], sort_keys=True)
            ),
            "- Five-run output consistent: `{}`".format(
                summary["stability"]["output_consistent"]
            ),
            "- Best configuration: `{}`".format(
                json.dumps(summary["best_configuration"], sort_keys=True)
            ),
            "- Decision: {}".format(summary["best_configuration_reason"]),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "reports" / "llama70b_single_gpu_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "llama70b_single_gpu_benchmark",
    )
    args = parser.parse_args(argv)
    rows = collect(args.input_root)
    from datetime import datetime, timezone

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = build_summary(rows, generated_at)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matrix.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render(rows, summary), encoding="utf-8"
    )
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
