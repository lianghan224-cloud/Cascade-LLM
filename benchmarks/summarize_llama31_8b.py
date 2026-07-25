#!/usr/bin/env python3
"""Validate and summarize the two-GPU Llama-3.1-8B M=1 benchmark."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


LAYER_BYTES = 436_224_000
TOTAL_LAYERS = 32
MEASURED_GROUPS = (1, 2, 4, 8)
ALL_GROUPS = (1, 2, 4, 8, 16, 32)
GIB = 1024**3


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/llama31_8b_m1_summary.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    pipeline = {}
    driver = {}
    checks = []
    sources = []

    def check(name, condition, detail):
        checks.append(
            {"name": name, "passed": bool(condition), "detail": detail}
        )

    for gpu in (0, 1):
        pipeline_path = root / "results/llama31_8b_m1_gpu{}.json".format(gpu)
        driver_path = (
            root / "results/h2d_llama31_8b_shards_gpu{}.json".format(gpu)
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

    pindex = {}
    dindex = {}
    for gpu in (0, 1):
        data = pipeline[gpu]
        check(
            "gpu{}_model_shape".format(gpu),
            data["model"]["id"] == "meta-llama/Llama-3.1-8B"
            and data["model"]["hidden_size"] == 4096
            and data["model"]["intermediate_size"] == 14336
            and data["model"]["num_hidden_layers"] == TOTAL_LAYERS
            and data["model"]["layer_bytes_with_norms"] == LAYER_BYTES,
            "exact Llama-3.1-8B decoder projection shape is declared",
        )
        pindex[gpu] = {
            group["group_layers"]: group for group in data["groups"]
        }
        dindex[gpu] = {
            row["size_bytes"] // LAYER_BYTES: row
            for row in driver[gpu]["transfer_results"]
        }
        check(
            "gpu{}_group_set".format(gpu),
            set(pindex[gpu]) == set(ALL_GROUPS),
            "all requested shard sizes have measured or infeasible rows",
        )
        for group in MEASURED_GROUPS:
            row = pindex[gpu][group]
            sequential = row["modes"]["sequential"]
            check(
                "gpu{}_g{}_measured".format(gpu, group),
                row["status"] == "measured"
                and sequential["total_ms"]["samples"] == 5,
                "five measured samples are present",
            )
            copy_ms = sequential["copy_total_ms"]["median"]
            check(
                "gpu{}_g{}_h2d_recompute".format(gpu, group),
                math.isclose(
                    row["metrics"]["h2d_total_GBps"],
                    TOTAL_LAYERS
                    * LAYER_BYTES
                    / (copy_ms * 1e6),
                    rel_tol=1e-12,
                ),
                "framework H2D throughput recomputes from bytes and time",
            )
        for group in (16, 32):
            check(
                "gpu{}_g{}_arena_infeasible".format(gpu, group),
                pindex[gpu][group]["status"] == "infeasible_for_arena",
                "two slots exceed the configured 8 GiB arena",
            )
        check(
            "gpu{}_driver_profile".format(gpu),
            driver[gpu]["benchmark"]
            == "cuda_driver_h2d_llama31_8b_shards"
            and driver[gpu]["profile"]
            == "llama31_8b_exact_layer_multiples_pinned_async",
            "raw Driver API result has the expected profile",
        )

    rows = []
    for group in MEASURED_GROUPS:
        shard_count = TOTAL_LAYERS // group
        gpu_rows = []
        for gpu in (0, 1):
            prow = pindex[gpu][group]
            sequential = prow["modes"]["sequential"]
            overlap = prow["modes"]["overlap"]
            drow = dindex[gpu][group]
            gpu_rows.append(
                {
                    "raw_driver_h2d_total_ms": (
                        shard_count
                        * drow["gpu_event_us"]["median"]
                        / 1000.0
                    ),
                    "raw_driver_h2d_GBps": drow[
                        "gpu_effective_gbps"
                    ]["median"],
                    "raw_driver_call_total_us": (
                        shard_count * drow["host_call_us"]["median"]
                    ),
                    "framework_h2d_total_ms": sequential["copy_total_ms"][
                        "median"
                    ],
                    "framework_h2d_GBps": prow["metrics"][
                        "h2d_total_GBps"
                    ],
                    "compute_total_ms": sequential["compute_total_ms"][
                        "median"
                    ],
                    "compute_total_TFLOPs": prow["metrics"][
                        "compute_total_TFLOPs"
                    ],
                    "torch_copy_submission_total_us": sequential[
                        "copy_submission_host_total_us"
                    ]["median"],
                    "full_host_enqueue_us": sequential[
                        "full_host_enqueue_us"
                    ]["median"],
                    "serial_ms": sequential["total_ms"]["median"],
                    "overlap_ms": overlap["total_ms"]["median"],
                    "speedup": prow["metrics"]["speedup_vs_sequential"],
                }
            )

        def mean(name):
            return statistics.fmean(item[name] for item in gpu_rows)

        h2d_ms = mean("framework_h2d_total_ms")
        compute_ms = mean("compute_total_ms")
        serial_ms = mean("serial_ms")
        overlap_ms = mean("overlap_ms")
        rows.append(
            {
                "status": "measured",
                "layers_per_shard": group,
                "shard_count": shard_count,
                "shard_MiB": group * LAYER_BYTES / (1024**2),
                "two_slot_GiB": 2 * group * LAYER_BYTES / GIB,
                "two_gpu_mean_of_medians": {
                    "raw_driver_h2d_total_ms": mean(
                        "raw_driver_h2d_total_ms"
                    ),
                    "raw_driver_h2d_GBps": mean(
                        "raw_driver_h2d_GBps"
                    ),
                    "raw_driver_call_total_us": mean(
                        "raw_driver_call_total_us"
                    ),
                    "framework_h2d_total_ms": h2d_ms,
                    "framework_h2d_GBps": mean("framework_h2d_GBps"),
                    "compute_total_ms": compute_ms,
                    "compute_total_TFLOPs": mean(
                        "compute_total_TFLOPs"
                    ),
                    "torch_copy_submission_total_us": mean(
                        "torch_copy_submission_total_us"
                    ),
                    "full_host_enqueue_us": mean("full_host_enqueue_us"),
                    "serial_ms": serial_ms,
                    "overlap_ms": overlap_ms,
                    "speedup": mean("speedup"),
                    "saved_by_overlap_ms": serial_ms - overlap_ms,
                },
                "derived": {
                    "h2d_to_compute_ratio": h2d_ms / compute_ms,
                    "h2d_fraction_of_serial": h2d_ms / serial_ms,
                    "compute_fraction_of_serial": compute_ms / serial_ms,
                    "component_optimistic_speedup": (
                        (h2d_ms + compute_ms) / max(h2d_ms, compute_ms)
                    ),
                },
                "per_gpu": {"gpu0": gpu_rows[0], "gpu1": gpu_rows[1]},
            }
        )

    for group in (16, 32):
        shard_bytes = group * LAYER_BYTES
        row = {
            "status": "infeasible_for_8GiB_double_buffer",
            "layers_per_shard": group,
            "shard_count": TOTAL_LAYERS // group,
            "shard_MiB": shard_bytes / (1024**2),
            "two_slot_GiB": 2 * shard_bytes / GIB,
        }
        if group == 16:
            raw_times = []
            raw_bandwidths = []
            raw_calls = []
            for gpu in (0, 1):
                drow = dindex[gpu][group]
                raw_times.append(
                    2 * drow["gpu_event_us"]["median"] / 1000.0
                )
                raw_bandwidths.append(
                    drow["gpu_effective_gbps"]["median"]
                )
                raw_calls.append(2 * drow["host_call_us"]["median"])
            row["single_buffer_raw_driver_projection"] = {
                "h2d_total_ms": statistics.fmean(raw_times),
                "h2d_GBps": statistics.fmean(raw_bandwidths),
                "driver_call_total_us": statistics.fmean(raw_calls),
            }
        rows.append(row)

    failed = [item for item in checks if not item["passed"]]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "llama31_8b_M1_two_gpu_summary",
        "assessment": "share_with_caveats" if not failed else "needs_revision",
        "scope": {
            "model": "meta-llama/Llama-3.1-8B",
            "dtype": "torch.bfloat16",
            "rows_M": 1,
            "arena_GiB": 8,
            "decoder_layers": TOTAL_LAYERS,
            "decoder_stream_bytes": TOTAL_LAYERS * LAYER_BYTES,
            "decoder_stream_GiB": TOTAL_LAYERS * LAYER_BYTES / GIB,
            "embedding_and_lm_head_excluded": True,
        },
        "source_files": sources,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": failed,
        "rows": rows,
        "caveats": [
            "Synthetic zero BF16 projection weights use exact Llama-3.1-8B shapes; official checkpoint tensors were not downloaded.",
            "Projection compute excludes attention score/softmax, RoPE, RMSNorm, KV cache, residuals, embedding, and LM head.",
            "The host source group is reused across synthetic stages; every transfer still moves the full requested bytes and the smallest group exceeds this CPU's LLC.",
            "Two-GPU values are means of per-GPU medians, not pooled samples.",
        ],
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "assessment": result["assessment"],
                "checks_passed": result["checks_passed"],
                "checks_total": result["checks_total"],
                "output": str(output),
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
