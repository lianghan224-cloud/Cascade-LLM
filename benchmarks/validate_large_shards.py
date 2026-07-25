#!/usr/bin/env python3
"""Validate and summarize the two-GPU large-shard benchmark results."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


LAYER_BYTES = 121_643_008
H2D_LAYER_MULTIPLES = (1, 2, 4, 8, 16)
PIPELINE_GROUPS = (1, 2, 4, 8)
ROWS = (1, 512, 2048, 4096)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_difference(left, right):
    return abs(left - right) / ((left + right) / 2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/large_shards_summary.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    checks = []

    def check(name, condition, detail):
        checks.append(
            {"name": name, "passed": bool(condition), "detail": detail}
        )

    h2d = {}
    pipeline = {}
    source_files = []
    for gpu in (0, 1):
        h2d_path = root / "results/h2d_large_shards_gpu{}.json".format(gpu)
        pipeline_path = (
            root / "results/large_shard_pipeline_gpu{}.json".format(gpu)
        )
        h2d[gpu] = json.loads(h2d_path.read_text())
        pipeline[gpu] = json.loads(pipeline_path.read_text())
        for path in (h2d_path, pipeline_path):
            source_files.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256(path),
                }
            )

    expected_sizes = tuple(
        multiple * LAYER_BYTES for multiple in H2D_LAYER_MULTIPLES
    )
    for gpu, data in h2d.items():
        rows = data["transfer_results"]
        sizes = tuple(row["size_bytes"] for row in rows)
        check(
            "gpu{}_h2d_profile".format(gpu),
            data["benchmark"] == "cuda_driver_h2d_large_shards"
            and data["profile"] == "exact_layer_multiples_pinned_async",
            "large-shard Driver API profile is explicitly identified",
        )
        check(
            "gpu{}_h2d_sizes".format(gpu),
            sizes == expected_sizes,
            "exact 1/2/4/8/16-layer byte multiples are present",
        )
        for multiple, row in zip(H2D_LAYER_MULTIPLES, rows):
            check(
                "gpu{}_h2d_{}layer_mode".format(gpu, multiple),
                row["memory"] == "pinned"
                and row["api"] == "cuMemcpyHtoDAsync_v2"
                and row["iterations"] == 10
                and row["warmups"] == 3,
                "pinned async H2D has 3 warmups and 10 measured samples",
            )
            for metric in ("gpu_event_us", "gpu_effective_gbps"):
                values = row[metric]
                check(
                    "gpu{}_h2d_{}layer_{}_ordering".format(
                        gpu, multiple, metric
                    ),
                    values["p10"] <= values["median"] <= values["p90"],
                    "p10 <= median <= p90",
                )
        bandwidths = [
            row["gpu_effective_gbps"]["median"] for row in rows
        ]
        bandwidth_spread = max(bandwidths) / min(bandwidths) - 1.0
        check(
            "gpu{}_h2d_bandwidth_plateau".format(gpu),
            bandwidth_spread < 0.01,
            "max/min median bandwidth spread is {:.3%}".format(
                bandwidth_spread
            ),
        )

    h2d_summary = []
    for index, multiple in enumerate(H2D_LAYER_MULTIPLES):
        gpu_rows = [
            h2d[gpu]["transfer_results"][index] for gpu in (0, 1)
        ]
        times_ms = [
            row["gpu_event_us"]["median"] / 1000.0 for row in gpu_rows
        ]
        bandwidths = [
            row["gpu_effective_gbps"]["median"] for row in gpu_rows
        ]
        host_calls = [row["host_call_us"]["median"] for row in gpu_rows]
        cross_gpu_difference = relative_difference(*bandwidths)
        check(
            "h2d_{}layer_two_gpu_reproducibility".format(multiple),
            cross_gpu_difference < 0.01,
            "median bandwidth differs by {:.3%}".format(
                cross_gpu_difference
            ),
        )
        h2d_summary.append(
            {
                "layers_per_shard": multiple,
                "bytes": multiple * LAYER_BYTES,
                "MiB": multiple * LAYER_BYTES / (1024.0 * 1024.0),
                "mean_of_gpu_median_time_ms": statistics.fmean(times_ms),
                "mean_of_gpu_median_GBps": statistics.fmean(bandwidths),
                "mean_of_gpu_median_host_call_us": statistics.fmean(
                    host_calls
                ),
                "gpu0_median_GBps": bandwidths[0],
                "gpu1_median_GBps": bandwidths[1],
            }
        )

    indexed_pipeline = {}
    for gpu, data in pipeline.items():
        groups = data["pipeline"]["groups"]
        indexed_pipeline[gpu] = {
            group["group_layers"]: {
                row["rows_M"]: row for row in group["rows"]
            }
            for group in groups
        }
        check(
            "gpu{}_pipeline_group_set".format(gpu),
            set(indexed_pipeline[gpu]) == set(PIPELINE_GROUPS),
            "1/2/4/8-layer groups are present regardless of run order",
        )
        check(
            "gpu{}_full_distinct_cpu_arena".format(gpu),
            data["cpu_weight_arena"]["pinned"]
            and data["cpu_weight_arena"]["distinct_layer_slices"]
            and data["cpu_weight_arena"]["bytes"]
            == 16 * LAYER_BYTES,
            "full 16-layer pinned arena uses distinct source slices",
        )
        for group in PIPELINE_GROUPS:
            group_data = next(
                item
                for item in groups
                if item["group_layers"] == group
            )
            check(
                "gpu{}_g{}_slot_bytes".format(gpu, group),
                group_data["slot_bytes"] == group * LAYER_BYTES
                and group_data["two_slot_bytes"] == 2 * group * LAYER_BYTES,
                "reported slot storage matches exact layer bytes",
            )
            check(
                "gpu{}_g{}_row_set".format(gpu, group),
                set(indexed_pipeline[gpu][group]) == set(ROWS),
                "all requested M values are present",
            )
            for rows_m in ROWS:
                row = indexed_pipeline[gpu][group][rows_m]
                sequential = row["modes"]["sequential"]["total_ms"]["median"]
                overlap = row["modes"]["overlap"]["total_ms"]["median"]
                check(
                    "gpu{}_g{}_M{}_speedup_recompute".format(
                        gpu, group, rows_m
                    ),
                    math.isclose(
                        row["speedup_vs_sequential"],
                        sequential / overlap,
                        rel_tol=1e-12,
                    ),
                    "saved speedup equals sequential / overlap",
                )
                check(
                    "gpu{}_g{}_M{}_direction".format(gpu, group, rows_m),
                    overlap < sequential,
                    "overlap is faster than the matched sequential baseline",
                )
                check(
                    "gpu{}_g{}_M{}_sample_count".format(
                        gpu, group, rows_m
                    ),
                    row["modes"]["sequential"]["total_ms"]["samples"] == 7
                    and row["modes"]["overlap"]["total_ms"]["samples"] == 7,
                    "both modes contain seven measured samples",
                )
                baseline_overlap = indexed_pipeline[gpu][1][rows_m][
                    "modes"
                ]["overlap"]["total_ms"]["median"]
                check(
                    "gpu{}_g{}_M{}_relative_throughput".format(
                        gpu, group, rows_m
                    ),
                    math.isclose(
                        row["overlap_throughput_relative_to_g1"],
                        baseline_overlap / overlap,
                        rel_tol=1e-12,
                    ),
                    "relative throughput is recomputed against g=1",
                )
        for rows_m in ROWS:
            speedups = [
                indexed_pipeline[gpu][group][rows_m][
                    "speedup_vs_sequential"
                ]
                for group in PIPELINE_GROUPS
            ]
            check(
                "gpu{}_M{}_speedup_decreases_with_group".format(gpu, rows_m),
                all(
                    speedups[index] > speedups[index + 1]
                    for index in range(len(speedups) - 1)
                ),
                "observed speedup strictly decreases for g=1,2,4,8",
            )
            sequential_times = [
                indexed_pipeline[gpu][group][rows_m]["modes"][
                    "sequential"
                ]["total_ms"]["median"]
                for group in PIPELINE_GROUPS
            ]
            spread = max(sequential_times) / min(sequential_times) - 1.0
            check(
                "gpu{}_M{}_constant_work_baseline".format(gpu, rows_m),
                spread < 0.015,
                "sequential max/min latency spread is {:.3%}".format(spread),
            )

    pipeline_summary = []
    for rows_m in ROWS:
        for group in PIPELINE_GROUPS:
            rows = [
                indexed_pipeline[gpu][group][rows_m] for gpu in (0, 1)
            ]
            sequential_times = [
                row["modes"]["sequential"]["total_ms"]["median"]
                for row in rows
            ]
            overlap_times = [
                row["modes"]["overlap"]["total_ms"]["median"]
                for row in rows
            ]
            speedups = [row["speedup_vs_sequential"] for row in rows]
            relative_throughputs = [
                row["overlap_throughput_relative_to_g1"] for row in rows
            ]
            cross_gpu_difference = relative_difference(*overlap_times)
            check(
                "g{}_M{}_two_gpu_reproducibility".format(group, rows_m),
                cross_gpu_difference < 0.01,
                "overlap latency differs by {:.3%}".format(
                    cross_gpu_difference
                ),
            )
            pipeline_summary.append(
                {
                    "rows_M": rows_m,
                    "layers_per_shard": group,
                    "shard_MiB": group * LAYER_BYTES / (1024.0 * 1024.0),
                    "two_slot_MiB": (
                        2 * group * LAYER_BYTES / (1024.0 * 1024.0)
                    ),
                    "mean_of_gpu_median_sequential_ms": statistics.fmean(
                        sequential_times
                    ),
                    "mean_of_gpu_median_overlap_ms": statistics.fmean(
                        overlap_times
                    ),
                    "mean_speedup_vs_sequential": statistics.fmean(speedups),
                    "mean_throughput_relative_to_g1": statistics.fmean(
                        relative_throughputs
                    ),
                    "gpu0_overlap_ms": overlap_times[0],
                    "gpu1_overlap_ms": overlap_times[1],
                }
            )

    failed = [item for item in checks if not item["passed"]]
    receipt = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assessment": "share_with_caveats" if not failed else "needs_revision",
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "blockers": (
            []
            if not failed
            else [
                "At least one large-shard reconciliation or reproducibility "
                "check failed."
            ]
        ),
        "source_files": source_files,
        "h2d_two_gpu_mean": h2d_summary,
        "pipeline_two_gpu_mean": pipeline_summary,
        "checks": checks,
        "required_caveats": [
            "Synthetic zero BF16 weights use exact decoder projection shapes; this is not an end-to-end generation benchmark.",
            "Projection compute excludes attention score/softmax, RoPE, RMSNorm, KV-cache, residual, embedding, and LM-head work.",
            "GPU clocks were not locked, so matched modes and two-GPU agreement are more reliable than isolated tiny-M timings.",
            "Two-GPU means are means of per-GPU medians, not pooled raw samples.",
        ],
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "assessment": receipt["assessment"],
                "checks_passed": receipt["checks_passed"],
                "checks_total": receipt["checks_total"],
                "failed": failed,
                "output": str(output),
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
