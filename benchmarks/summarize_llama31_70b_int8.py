#!/usr/bin/env python3
"""Validate and summarize real Llama 3.1 70B W8A8 benchmark results."""

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "real_results" / "70b_int8"
GIB = 1024**3

FILES = {
    "full_pinned_matrix_s1": "bench_full_pinned_matrix_s1.json",
    "full_pinned_matrix_s2": "bench_full_pinned_matrix_s2_repeat.json",
    "full_pinned_layer_s2": "bench_full_pinned_layer_s2.json",
    "pinned_staging_matrix_s1": "bench_pinned_staging_matrix_s1.json",
    "pinned_staging_matrix_s2": "bench_pinned_staging_matrix_s2.json",
    "pinned_staging_matrix_group_s2": (
        "bench_pinned_staging_matrix_group_s2.json"
    ),
    "pinned_staging_layer_s2": (
        "bench_pinned_staging_layer_s2_repeat.json"
    ),
}


def median_step(report):
    # Keep the headline latency free of fine-grained CUDA profiling overhead.
    # Profile runs are summarized separately through their event timings.
    return statistics.median(report["decode_wall_ms"])


def median_profile(report, key):
    return statistics.median(row[key] for row in report["profiles"])


def optional_median_profile(report, key, default=0.0):
    values = [
        row[key] for row in report["profiles"] if key in row
    ]
    return statistics.median(values) if values else default


def median_vocab(report, key):
    return statistics.median(row[key] for row in report["vocab_profiles"])


def load_results():
    loaded = {}
    for name, filename in FILES.items():
        path = RESULTS / filename
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def main():
    loaded = load_results()
    validation = json.loads(
        (RESULTS / "checkpoint_validation.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for name, report in loaded.items():
        latency = median_step(report)
        compute_ms = median_profile(report, "compute_event_sum_ms")
        dequant_ms = median_profile(report, "dequant_event_sum_ms")
        pinned = (
            report["cpu_pinned_bytes"]
            + report["vocab_extra_pinned_cpu_bytes"]
        )
        rows.append(
            {
                "name": name,
                "weight_store": report["weight_store"],
                "granularity": report["granularity"],
                "slots": report["slots"],
                "transfer_units": report["runtime"]["transfer_units"],
                "median_token_ms": latency,
                "tokens_per_second": 1000.0 / latency,
                "transformer_h2d_ms": median_profile(
                    report, "h2d_event_sum_ms"
                ),
                "transformer_h2d_gbps": median_profile(
                    report, "h2d_effective_gbps"
                ),
                "transformer_compute_ms": compute_ms,
                "dequant_ms": dequant_ms,
                "compute_excluding_dequant_ms": compute_ms - dequant_ms,
                "transformer_pipeline_ms": median_profile(
                    report, "gpu_pipeline_ms"
                ),
                "lm_head_pipeline_ms": median_vocab(
                    report, "pipeline_ms"
                ),
                "source_wait_ms": median_profile(
                    report, "source_wait_ms"
                ),
                "host_submit_ms": median_profile(
                    report, "host_submit_ms"
                ),
                "staging_copy_event_sum_ms": optional_median_profile(
                    report, "staging_event_sum_ms"
                ),
                "staging_copy_event_max_ms": optional_median_profile(
                    report, "staging_event_max_ms"
                ),
                "peak_gpu_allocated_bytes": report[
                    "cuda_peak_allocated_bytes"
                ],
                "planned_weight_gpu_bytes": report["runtime"][
                    "weight_gpu_bytes"
                ],
                "pinned_cpu_bytes": pinned,
                "cpu_arena_pinned": report["cpu_arena_is_pinned"],
                "allocation_seconds": report["allocation_seconds"],
                "load_seconds": report["load_seconds"],
                "generated_token_ids": report["generated_token_ids"],
            }
        )

    by_name = {row["name"]: row for row in rows}
    full_s1 = by_name["full_pinned_matrix_s1"]
    full_s2 = by_name["full_pinned_matrix_s2"]
    staging_s1 = by_name["pinned_staging_matrix_s1"]
    staging_s2 = by_name["pinned_staging_matrix_s2"]
    staging_best = by_name["pinned_staging_layer_s2"]
    checkpoint_bytes = validation["tensor_payload_bytes"]
    full_repeat = loaded["full_pinned_matrix_s2"]
    staging_repeat = loaded["pinned_staging_layer_s2"]

    topk_equal = (
        full_repeat.get("generated_topk")
        == staging_repeat.get("generated_topk")
    )
    generated_equal = (
        full_repeat["generated_token_ids"]
        == staging_repeat["generated_token_ids"]
    )
    total_h2d_ms = (
        full_s2["transformer_h2d_ms"]
        + median_vocab(full_repeat, "h2d_event_sum_ms")
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_validation": validation,
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 3080 Ti 12 GiB",
            "gpu_count_used": 1,
            "cpu": "AMD Ryzen Threadripper 3970X 32-Core",
            "system_ram_gib": 125,
        },
        "measurement": {
            "prompt": full_repeat["prompt"],
            "prompt_tokens": full_repeat["prompt_tokens"],
            "decode_scope": "single request, batch=1, one-token decode",
            "latency_statistic": (
                "median of decode_wall_ms without fine-grained profiling"
            ),
            "weight_bytes_per_decode": full_repeat[
                "h2d_accounting"
            ]["total_decode_weight_bytes"],
            "profile_note": (
                "Transformer H2D/compute events exclude streamed LM-head; "
                "LM-head timings are reported separately."
            ),
        },
        "benchmark_rows": rows,
        "comparisons": {
            "full_pinned_double_vs_single_speedup": (
                full_s1["median_token_ms"] / full_s2["median_token_ms"]
            ),
            "staging_matrix_double_vs_single_speedup": (
                staging_s1["median_token_ms"]
                / staging_s2["median_token_ms"]
            ),
            "staging_best_vs_matrix_single_speedup": (
                staging_s1["median_token_ms"]
                / staging_best["median_token_ms"]
            ),
            "full_pinned_best_vs_staging_best_speedup": (
                staging_best["median_token_ms"]
                / full_s2["median_token_ms"]
            ),
            "full_pinned_gpu_memory_reduction_ratio_vs_int8_resident": (
                checkpoint_bytes / full_s2["peak_gpu_allocated_bytes"]
            ),
            "full_pinned_gpu_memory_saved_fraction_vs_int8_resident": (
                1.0
                - full_s2["peak_gpu_allocated_bytes"] / checkpoint_bytes
            ),
            "staging_gpu_memory_reduction_ratio_vs_int8_resident": (
                checkpoint_bytes
                / staging_best["peak_gpu_allocated_bytes"]
            ),
            "double_buffer_hidden_compute_fraction": (
                (
                    full_s1["transformer_pipeline_ms"]
                    - full_s2["transformer_pipeline_ms"]
                )
                / full_s1["transformer_compute_ms"]
            ),
            "full_pinned_total_h2d_ms": total_h2d_ms,
            "full_pinned_h2d_fraction_of_token_wall": (
                total_h2d_ms / full_s2["median_token_ms"]
            ),
        },
        "correctness": {
            "generated_tokens_equal_between_recommended_modes": (
                generated_equal
            ),
            "topk_equal_between_recommended_modes": topk_equal,
            "generated_token_ids": full_repeat["generated_token_ids"],
            "generated_text": full_repeat["generated_text"],
            "external_compressed_tensors_reference": "not_run",
            "external_reference_reason": (
                "compressed_tensors is not installed; cross-implementation "
                "validation remains required before claiming model-level "
                "numerical equivalence."
            ),
        },
        "caveats": [
            (
                "The runtime transfers checkpoint INT8 weights and "
                "dequantizes to BF16 on GPU; it does not execute W8A8 "
                "activation-quantized kernels."
            ),
            (
                "The full-resident 70B INT8 baseline is infeasible on a "
                "12 GiB GPU; memory ratios use the exact checkpoint tensor "
                "payload as the minimum full-resident weight requirement."
            ),
            (
                "Performance is batch=1 decode on one RTX 3080 Ti and should "
                "not be generalized to prefill, multi-request serving, or "
                "different PCIe/topology without remeasurement."
            ),
        ],
    }
    output = RESULTS / "summary.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    checks = [
        validation["status"] == "passed",
        generated_equal,
        topk_equal,
        full_s2["median_token_ms"] < full_s1["median_token_ms"],
        staging_best["median_token_ms"]
        < staging_s1["median_token_ms"],
        full_s2["source_wait_ms"] < 10.0,
        full_s2["transformer_h2d_gbps"] > 23.0,
    ]
    if not all(checks):
        raise SystemExit("summary validation failed")


if __name__ == "__main__":
    main()
