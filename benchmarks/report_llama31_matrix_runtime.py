#!/usr/bin/env python3
"""Create an auditable memory/performance estimate from the measured 8B baseline."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import build_llama31_8b_plan  # noqa: E402


MIB = 1024 * 1024
GIB = 1024 * MIB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/llama31_8b_m1_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/llama31_8b_matrix_runtime_report.json"),
    )
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    measured = baseline["rows"][0]["two_gpu_mean_of_medians"]
    layer_plan = build_llama31_8b_plan("layer")
    matrix_plan = build_llama31_8b_plan("matrix")

    raw_submit_per_copy_us = (
        measured["raw_driver_call_total_us"] / 32.0
    )
    framework_submit_per_copy_us = (
        measured["torch_copy_submission_total_us"] / 32.0
    )
    extra_copies = len(matrix_plan.units) - len(layer_plan.units)
    raw_extra_ms = extra_copies * raw_submit_per_copy_us / 1000.0
    framework_extra_ms = (
        extra_copies * framework_submit_per_copy_us / 1000.0
    )
    layer_overlap_ms = measured["overlap_ms"]
    serial_ms = measured["serial_ms"]
    single_thread_stage_gbps = 14.8
    staging_single_thread_ms = (
        matrix_plan.stream_bytes_per_token
        / (single_thread_stage_gbps * 1e9)
        * 1000.0
    )
    required_staging_gbps = (
        matrix_plan.stream_bytes_per_token
        / (layer_overlap_ms / 1000.0)
        / 1e9
    )

    result = {
        "schema_version": 1,
        "model": matrix_plan.model_id,
        "workload": "BF16 M=1 decoder projections; embedding/lm_head excluded from timing",
        "status": "modeled_from_measured_baseline",
        "baseline_source": str(args.baseline),
        "baseline": {
            "granularity": "transformer_layer",
            "transfer_units": len(layer_plan.units),
            "two_slot_bytes": layer_plan.two_slot_bytes,
            "resident_bytes": layer_plan.resident_bytes,
            "total_weight_gpu_bytes": (
                layer_plan.two_slot_bytes + layer_plan.resident_bytes
            ),
            "serial_ms": serial_ms,
            "overlap_ms": layer_overlap_ms,
            "speedup_vs_serial": serial_ms / layer_overlap_ms,
        },
        "matrix_runtime": {
            "granularity": "projection_matrix",
            "transfer_units": len(matrix_plan.units),
            "max_matrix_bytes": matrix_plan.slot_bytes,
            "two_slot_bytes": matrix_plan.two_slot_bytes,
            "resident_bytes": matrix_plan.resident_bytes,
            "total_weight_gpu_bytes": (
                matrix_plan.two_slot_bytes + matrix_plan.resident_bytes
            ),
            "full_pinned_cpu_bytes": matrix_plan.host_arena_bytes,
            "pinned_staging_cpu_bytes": matrix_plan.two_slot_bytes,
            "pinned_staging_total_cpu_bytes": (
                matrix_plan.host_arena_bytes
                + matrix_plan.two_slot_bytes
            ),
        },
        "memory_improvement": {
            "saved_bytes": (
                layer_plan.two_slot_bytes - matrix_plan.two_slot_bytes
            ),
            "stream_buffer_reduction_fraction": (
                1.0
                - matrix_plan.two_slot_bytes / layer_plan.two_slot_bytes
            ),
            "total_weight_gpu_reduction_fraction": (
                1.0
                - (
                    matrix_plan.two_slot_bytes
                    + matrix_plan.resident_bytes
                )
                / (
                    layer_plan.two_slot_bytes
                    + layer_plan.resident_bytes
                )
            ),
        },
        "performance_estimate": {
            "h2d_payload_unchanged_bytes": (
                matrix_plan.stream_bytes_per_token
            ),
            "extra_copy_submissions": extra_copies,
            "raw_driver_extra_ms": raw_extra_ms,
            "framework_submission_extra_ms": framework_extra_ms,
            "optimistic_matrix_overlap_ms": (
                layer_overlap_ms + raw_extra_ms
            ),
            "conservative_matrix_overlap_ms": (
                layer_overlap_ms + framework_extra_ms
            ),
            "optimistic_speedup_vs_serial": (
                serial_ms / (layer_overlap_ms + raw_extra_ms)
            ),
            "conservative_speedup_vs_serial": (
                serial_ms / (layer_overlap_ms + framework_extra_ms)
            ),
            "optimistic_ratio_vs_layer_overlap": (
                layer_overlap_ms / (layer_overlap_ms + raw_extra_ms)
            ),
            "conservative_ratio_vs_layer_overlap": (
                layer_overlap_ms
                / (layer_overlap_ms + framework_extra_ms)
            ),
            "pinned_staging": {
                "measured_single_thread_memcpy_GBps": (
                    single_thread_stage_gbps
                ),
                "single_thread_staging_ms_per_token": (
                    staging_single_thread_ms
                ),
                "aggregate_memcpy_GBps_needed_to_avoid_throttling_h2d": (
                    required_staging_gbps
                ),
                "default_cpu_workers": 2,
                "assessment": (
                    "requires measurement; one measured CPU worker is "
                    "insufficient, two workers can only preserve full-pinned "
                    "speed if aggregate staging exceeds the required rate"
                ),
            },
        },
        "caveats": [
            "Performance values are projections from saved measurements, not a new GPU run.",
            "The baseline uses exact synthetic projection shapes and excludes embedding/lm_head.",
            "Matrix transfers may have size-dependent bandwidth and require a new CUDA measurement.",
            "Pinned-staging also adds pageable-to-pinned CPU memcpy, which is not included in the latency estimate.",
            "The framework submission projection does not include every extra Python callback and compute-stream event.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
