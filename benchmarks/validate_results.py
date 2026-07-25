#!/usr/bin/env python3
"""Recompute the high-impact benchmark claims and emit a QA receipt."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


LAYER_BYTES = 121_643_008
MODEL_TENSOR_BYTES = 2_471_628_800
PIPELINE_ROWS = (1, 8, 32, 128, 512, 1024, 2048, 4096)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, rel=1e-6, abs_tol=1e-9):
    return math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    checks = []

    def check(name, condition, detail):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    h2d_summary = json.loads((root / "h2d_results_summary.json").read_text())
    raw_h2d = {}
    for gpu in (0, 1):
        path = root / "h2d_results_gpu{}.json".format(gpu)
        raw_h2d[gpu] = json.loads(path.read_text())
        declared = next(
            item
            for item in h2d_summary["generated_from"]
            if item["path"] == path.name
        )
        check(
            "gpu{}_raw_sha256".format(gpu),
            sha256(path) == declared["sha256"],
            "raw result matches h2d_results_summary.json",
        )

        exact = next(
            row
            for row in raw_h2d[gpu]["transfer_results"]
            if row["memory"] == "pinned"
            and row["api"] == "cuMemcpyHtoDAsync_v2"
            and row["size_bytes"] == LAYER_BYTES
        )
        compact = next(
            row
            for row in h2d_summary[
                "pinned_async_model_relevant_medians"
            ]["gpu{}".format(gpu)]
            if row["size"] == LAYER_BYTES
        )
        check(
            "gpu{}_exact_layer_h2d_reconciliation".format(gpu),
            close(exact["gpu_event_us"]["median"], compact["gpu_event"])
            and close(
                exact["gpu_effective_gbps"]["median"],
                compact["bandwidth"],
            )
            and close(exact["host_call_us"]["median"], compact["host_call"]),
            "raw and compact exact-layer medians agree",
        )

    torch_results = {}
    for gpu in (0, 1):
        path = root / "results/torch_llama32_1b_gpu{}.json".format(gpu)
        data = json.loads(path.read_text())
        torch_results[gpu] = data
        check(
            "gpu{}_model_shape".format(gpu),
            data["model"]["layer_bytes_with_norms"] == LAYER_BYTES,
            "exact layer payload is 121,643,008 bytes",
        )
        check(
            "gpu{}_pipeline_rows".format(gpu),
            tuple(row["rows_M"] for row in data["pipeline"]["rows"])
            == PIPELINE_ROWS,
            "all requested M values are present in order",
        )
        for row in data["pipeline"]["rows"]:
            sequential = row["modes"]["sequential"]["total_ms"]["median"]
            overlap = row["modes"]["overlap"]["total_ms"]["median"]
            check(
                "gpu{}_M{}_pipeline_speedup".format(gpu, row["rows_M"]),
                close(row["speedup_vs_sequential"], sequential / overlap),
                "saved speedup equals sequential_ms / overlap_ms",
            )
            check(
                "gpu{}_M{}_pipeline_direction".format(gpu, row["rows_M"]),
                overlap < sequential,
                "overlap is faster than the matched sequential baseline",
            )
        for row in data["overlap"]["rows"]:
            modes = row["modes"]
            copy_span = modes["isolated_copy"]["span_ms"]["median"]
            compute_span = modes["isolated_compute"]["span_ms"]["median"]
            overlap_span = modes["overlap"]["span_ms"]["median"]
            raw_hidden = (
                copy_span + compute_span - overlap_span
            ) / min(copy_span, compute_span)
            check(
                "gpu{}_M{}_hidden_scope".format(gpu, row["rows_M"]),
                close(
                    row["metrics"]["hidden_fraction_raw_estimate"],
                    raw_hidden,
                )
                and close(
                    row["metrics"]["hidden_fraction"],
                    min(1.0, raw_hidden),
                ),
                "hidden fraction uses common start-to-finish spans",
            )

    for m in PIPELINE_ROWS:
        rows = [
            next(
                row
                for row in torch_results[gpu]["pipeline"]["rows"]
                if row["rows_M"] == m
            )
            for gpu in (0, 1)
        ]
        relative_difference = abs(
            rows[0]["speedup_vs_sequential"]
            - rows[1]["speedup_vs_sequential"]
        ) / (
            (
                rows[0]["speedup_vs_sequential"]
                + rows[1]["speedup_vs_sequential"]
            )
            / 2
        )
        check(
            "M{}_two_gpu_reproducibility".format(m),
            relative_difference < 0.02,
            "pipeline speedups differ by {:.3%}".format(relative_difference),
        )

    arena = json.loads(
        (root / "results/pinned_arena_probe.json").read_text()
    )
    check(
        "full_pinned_arena",
        arena["status"] == "ok"
        and arena["requested_bytes"] == MODEL_TENSOR_BYTES
        and arena["allocation_api"] == "cuMemHostAlloc",
        "full 2,471,628,800-byte tensor payload was allocated with CUDA Driver API",
    )

    failed = [item for item in checks if not item["passed"]]
    receipt = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "machine calibration and synthetic exact-shape projection/overlap "
            "benchmark"
        ),
        "assessment": "share_with_caveats" if not failed else "needs_revision",
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "checks": checks,
        "blockers": (
            []
            if not failed
            else [
                "One or more saved-result reconciliation checks failed; do not "
                "share derived benchmark claims until fixed."
            ]
        ),
        "required_caveats": [
            "The official model weights are manual-gated and were not available on this host.",
            "GPU compute is exact-shape BF16 projection-only, not a full decoder or end-to-end generation benchmark.",
            "GPU clocks were not administratively locked; matched sequential/overlap comparisons are more reliable than tiny-M component slowdown.",
            "Pipeline CUDA timing events add equal instrumentation to both modes; production absolute overhead will differ.",
        ],
    }
    output = args.output
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
