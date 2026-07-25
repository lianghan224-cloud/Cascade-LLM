#!/usr/bin/env python3
"""Summarize M=1 component accounting across two GPUs and shard sizes."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


LAYER_BYTES = 121_643_008
TOTAL_LAYERS = 16
GROUPS = (1, 2, 4, 8, 16)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/shard_accounting_m1_summary.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    pipeline = {}
    driver = {}
    sources = []
    checks = []

    def check(name, condition, detail):
        checks.append(
            {"name": name, "passed": bool(condition), "detail": detail}
        )

    for gpu in (0, 1):
        pipeline_path = (
            root / "results/shard_accounting_m1_gpu{}.json".format(gpu)
        )
        driver_path = (
            root / "results/h2d_large_shards_gpu{}.json".format(gpu)
        )
        pipeline[gpu] = json.loads(pipeline_path.read_text())
        driver[gpu] = json.loads(driver_path.read_text())
        for path in (pipeline_path, driver_path):
            sources.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256(path),
                }
            )

    indexed_pipeline = {}
    indexed_driver = {}
    for gpu in (0, 1):
        indexed_pipeline[gpu] = {
            group["group_layers"]: group["rows"][0]
            for group in pipeline[gpu]["pipeline"]["groups"]
        }
        indexed_driver[gpu] = {
            row["size_bytes"] // LAYER_BYTES: row
            for row in driver[gpu]["transfer_results"]
        }
        check(
            "gpu{}_group_set".format(gpu),
            set(indexed_pipeline[gpu]) == set(GROUPS),
            "all 1/2/4/8/16-layer shard sizes are present",
        )
        for group in GROUPS:
            row = indexed_pipeline[gpu][group]
            mode = row["modes"]["sequential"]
            check(
                "gpu{}_g{}_M1".format(gpu, group),
                row["rows_M"] == 1,
                "the accounting point is single-row M=1",
            )
            check(
                "gpu{}_g{}_samples".format(gpu, group),
                mode["total_ms"]["samples"] == 20,
                "twenty measured samples are present",
            )
            check(
                "gpu{}_g{}_component_identity".format(gpu, group),
                math.isclose(
                    row["sequential_component_H2D_GBps"],
                    TOTAL_LAYERS
                    * LAYER_BYTES
                    / (mode["copy_total_ms"]["median"] * 1e6),
                    rel_tol=1e-12,
                ),
                "saved H2D throughput recomputes from total bytes and time",
            )

    rows = []
    for group in GROUPS:
        shard_count = TOTAL_LAYERS // group
        gpu_values = []
        for gpu in (0, 1):
            row = indexed_pipeline[gpu][group]
            sequential = row["modes"]["sequential"]
            overlap = row["modes"]["overlap"]
            raw_driver = indexed_driver[gpu][group]
            gpu_values.append(
                {
                    "h2d_total_ms": sequential["copy_total_ms"]["median"],
                    "h2d_GBps": row["sequential_component_H2D_GBps"],
                    "compute_total_ms": sequential["compute_total_ms"][
                        "median"
                    ],
                    "compute_TFLOPs": row[
                        "sequential_component_projection_TFLOPs"
                    ],
                    "driver_h2d_call_total_us": (
                        shard_count
                        * raw_driver["host_call_us"]["median"]
                    ),
                    "torch_copy_call_total_us": sequential[
                        "copy_call_host_total_us"
                    ]["median"],
                    "torch_copy_submission_total_us": sequential[
                        "copy_submission_host_total_us"
                    ]["median"],
                    "torch_compute_submission_total_us": sequential[
                        "compute_submission_host_total_us"
                    ]["median"],
                    "full_host_enqueue_us": sequential["host_enqueue_us"][
                        "median"
                    ],
                    "serial_ms": sequential["total_ms"]["median"],
                    "overlap_ms": overlap["total_ms"]["median"],
                    "speedup": row["speedup_vs_sequential"],
                }
            )

        def mean(name):
            return statistics.fmean(value[name] for value in gpu_values)

        h2d_total_ms = mean("h2d_total_ms")
        compute_total_ms = mean("compute_total_ms")
        serial_ms = mean("serial_ms")
        overlap_ms = mean("overlap_ms")
        driver_total_us = mean("driver_h2d_call_total_us")
        torch_copy_submission_us = mean("torch_copy_submission_total_us")
        rows.append(
            {
                "layers_per_shard": group,
                "shard_count": shard_count,
                "shard_bytes": group * LAYER_BYTES,
                "shard_MiB": group
                * LAYER_BYTES
                / (1024.0 * 1024.0),
                "two_gpu_mean_of_medians": {
                    "h2d_total_ms": h2d_total_ms,
                    "h2d_GBps": mean("h2d_GBps"),
                    "compute_total_ms": compute_total_ms,
                    "compute_TFLOPs": mean("compute_TFLOPs"),
                    "driver_h2d_call_total_us": driver_total_us,
                    "torch_copy_call_total_us": mean(
                        "torch_copy_call_total_us"
                    ),
                    "torch_copy_submission_total_us": (
                        torch_copy_submission_us
                    ),
                    "torch_compute_submission_total_us": mean(
                        "torch_compute_submission_total_us"
                    ),
                    "full_host_enqueue_us": mean("full_host_enqueue_us"),
                    "serial_ms": serial_ms,
                    "overlap_ms": overlap_ms,
                    "speedup": mean("speedup"),
                    "saved_by_overlap_ms": serial_ms - overlap_ms,
                },
                "derived_shares": {
                    "h2d_fraction_of_serial": h2d_total_ms / serial_ms,
                    "compute_fraction_of_serial": (
                        compute_total_ms / serial_ms
                    ),
                    "driver_h2d_call_fraction_of_h2d": (
                        driver_total_us / (h2d_total_ms * 1000.0)
                    ),
                    "torch_copy_submission_fraction_of_h2d": (
                        torch_copy_submission_us
                        / (h2d_total_ms * 1000.0)
                    ),
                },
                "per_gpu": {
                    "gpu0": gpu_values[0],
                    "gpu1": gpu_values[1],
                },
            }
        )

    failed = [item for item in checks if not item["passed"]]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "M1_shard_component_accounting",
        "assessment": "share_with_caveats" if not failed else "needs_revision",
        "scope": {
            "rows_M": 1,
            "total_layers": TOTAL_LAYERS,
            "total_weight_bytes": TOTAL_LAYERS * LAYER_BYTES,
            "compute": "synthetic BF16 decoder projection-linear-only",
        },
        "definitions": {
            "h2d_total_ms": (
                "sum of per-shard CUDA Event H2D durations in the matched "
                "sequential pipeline sample"
            ),
            "compute_total_ms": (
                "sum of per-shard CUDA Event projection compute durations"
            ),
            "driver_h2d_call_total_us": (
                "raw Driver API median host call multiplied by shard count"
            ),
            "torch_copy_submission_total_us": (
                "host time for stream context, begin/end Event records, and "
                "Tensor.copy_ calls, summed across shards"
            ),
            "full_host_enqueue_us": (
                "host wall time to enqueue the full pass; asynchronous and "
                "therefore not additive to the GPU critical path"
            ),
        },
        "source_files": sources,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": failed,
        "rows": rows,
        "caveats": [
            "The compute scope excludes attention score/softmax, RoPE, RMSNorm, KV cache, residuals, embedding, and LM head.",
            "Host enqueue work is asynchronous and mostly overlaps the much longer DMA timeline; it must not be blindly added to CUDA Event component times.",
            "Two-GPU values are means of per-GPU medians, not pooled samples.",
        ],
    }
    output_path = (
        args.output if args.output.is_absolute() else root / args.output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "assessment": output["assessment"],
                "checks_passed": output["checks_passed"],
                "checks_total": output["checks_total"],
                "output": str(output_path),
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
