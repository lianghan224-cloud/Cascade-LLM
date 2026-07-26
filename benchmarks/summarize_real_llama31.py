#!/usr/bin/env python3
"""Summarize and validate real Llama-3.1-8B benchmark artifacts."""

import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "real_results"
CHECKPOINT = Path("/ssd/cascade-llm/models/Llama-3.1-8B")
GIB = 1024**3

BENCHMARKS = {
    "gpu0_full_pinned_layer_s1": "bench_full_pinned_layer_s1.json",
    "gpu0_full_pinned_matrix_s1": "bench_full_pinned_matrix_s1.json",
    "gpu0_full_pinned_layer_s2": "bench_full_pinned_layer_s2.json",
    "gpu0_full_pinned_matrix_s2": "bench_full_pinned_matrix_s2.json",
    "gpu0_pinned_staging_layer_s2": "bench_pinned_staging_layer_s2.json",
    "gpu0_pinned_staging_matrix_s2": "bench_pinned_staging_matrix_s2.json",
    "gpu1_full_pinned_layer_s1": "bench_gpu1_full_pinned_layer_s1.json",
    "gpu1_full_pinned_matrix_s2": "bench_gpu1_full_pinned_matrix_s2.json",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def median_profile(report, field):
    return statistics.median(item[field] for item in report["profiles"])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    return subprocess.check_output(
        ["git", "-C", str(CHECKPOINT), *args],
        text=True,
    ).strip()


def inspect_safetensors(paths):
    tensor_count = 0
    parameter_elements = 0
    payload_bytes = 0
    dtype_widths = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "F64": 8,
        "I8": 1,
        "U8": 1,
        "I16": 2,
        "U16": 2,
        "I32": 4,
        "U32": 4,
        "I64": 8,
        "U64": 8,
        "BOOL": 1,
    }
    dtypes = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as source:
            for key in source.keys():
                tensor_slice = source.get_slice(key)
                shape = tensor_slice.get_shape()
                dtype = tensor_slice.get_dtype()
                elements = 1
                for extent in shape:
                    elements *= extent
                tensor_count += 1
                parameter_elements += elements
                payload_bytes += elements * dtype_widths[dtype]
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
    return {
        "tensor_count": tensor_count,
        "parameter_elements": parameter_elements,
        "parameter_payload_bytes": payload_bytes,
        "tensor_dtype_counts": dtypes,
    }


def main():
    loaded = {}
    sources = []
    rows = []
    for key, filename in BENCHMARKS.items():
        path = RESULTS / filename
        report = load_json(path)
        loaded[key] = report
        sources.append(
            {
                "file": f"real_results/{filename}",
                "sha256": sha256(path),
            }
        )
        pipeline_ms = median_profile(report, "gpu_pipeline_ms")
        h2d_ms = median_profile(report, "h2d_event_sum_ms")
        compute_ms = median_profile(report, "compute_event_sum_ms")
        row = {
            "configuration": key,
            "gpu": int(key[3:].split("_", maxsplit=1)[0]),
            "weight_store": report["weight_store"],
            "granularity": report["granularity"],
            "slots": report["slots"],
            "transfer_units": report["runtime"]["transfer_units"],
            "decode_repeats": report["decode_repeats"],
            "decode_median_ms": report[
                "decode_wall_distribution_ms"
            ]["median"],
            "decode_p10_ms": report["decode_wall_distribution_ms"]["p10"],
            "decode_p90_ms": report["decode_wall_distribution_ms"]["p90"],
            "tokens_per_second": (
                1000.0
                / report["decode_wall_distribution_ms"]["median"]
            ),
            "profile_pipeline_median_ms": pipeline_ms,
            "profile_h2d_sum_median_ms": h2d_ms,
            "profile_compute_sum_median_ms": compute_ms,
            "profile_serial_h2d_share_pct": (
                100.0 * h2d_ms / (h2d_ms + compute_ms)
            ),
            "h2d_effective_gbps": median_profile(
                report, "h2d_effective_gbps"
            ),
            "host_submit_median_ms": median_profile(
                report, "host_submit_ms"
            ),
            "source_wait_median_ms": median_profile(
                report, "source_wait_ms"
            ),
            "streamed_weight_bytes_per_token": report["profiles"][0][
                "h2d_bytes"
            ],
            "cpu_pinned_bytes": report["cpu_pinned_bytes"],
            "cpu_arena_is_pinned": report.get("cpu_arena_is_pinned"),
            "gpu_planned_weight_bytes": report["runtime"][
                "weight_gpu_bytes"
            ],
            "gpu_peak_allocated_bytes": report[
                "cuda_peak_allocated_bytes"
            ],
            "gpu_peak_reserved_bytes": report.get(
                "cuda_peak_reserved_bytes"
            ),
            "measured_decode_disk_read_bytes": report[
                "measured_decode_io_delta"
            ]["read_bytes"],
        }
        if report["weight_store"] == "pinned_staging":
            row["staging_copy_sum_median_ms"] = median_profile(
                report, "staging_event_sum_ms"
            )
            row["staging_copy_count"] = report["profiles"][0][
                "staging_copy_count"
            ]
        rows.append(row)

    correctness_path = (
        RESULTS / "correctness_comparison_matrix_staging_8token.json"
    )
    correctness = load_json(correctness_path)
    sources.append(
        {
            "file": (
                "real_results/"
                "correctness_comparison_matrix_staging_8token.json"
            ),
            "sha256": sha256(correctness_path),
        }
    )

    primary = loaded["gpu0_full_pinned_matrix_s2"]
    layer_serial = loaded["gpu0_full_pinned_layer_s1"]
    matrix_serial = loaded["gpu0_full_pinned_matrix_s1"]
    layer_double = loaded["gpu0_full_pinned_layer_s2"]
    staging_matrix = loaded["gpu0_pinned_staging_matrix_s2"]
    gpu1_primary = loaded["gpu1_full_pinned_matrix_s2"]

    def decode_median(report):
        return report["decode_wall_distribution_ms"]["median"]

    parameter_payload_bytes = 16_060_522_496
    gpu_capacity_bytes = 12_528_123_904
    primary_gpu_bytes = primary["runtime"]["weight_gpu_bytes"]
    matrix_slot_bytes = primary["runtime"]["slot_bytes"]
    layer_slot_bytes = layer_double["runtime"]["slot_bytes"]
    comparisons = {
        "full_gpu_baseline_feasible": (
            parameter_payload_bytes <= gpu_capacity_bytes
        ),
        "parameter_payload_bytes": parameter_payload_bytes,
        "gpu_capacity_bytes": gpu_capacity_bytes,
        "parameter_payload_to_gpu_capacity_ratio": (
            parameter_payload_bytes / gpu_capacity_bytes
        ),
        "streaming_weight_memory_reduction_factor": (
            parameter_payload_bytes / primary_gpu_bytes
        ),
        "streaming_weight_memory_saved_pct": (
            100.0
            * (parameter_payload_bytes - primary_gpu_bytes)
            / parameter_payload_bytes
        ),
        "streaming_weight_memory_saved_bytes": (
            parameter_payload_bytes - primary_gpu_bytes
        ),
        "matrix_vs_layer_double_buffer_saved_bytes": (
            2 * (layer_slot_bytes - matrix_slot_bytes)
        ),
        "matrix_vs_layer_total_weight_gpu_saved_pct": (
            100.0
            * (
                layer_double["runtime"]["weight_gpu_bytes"]
                - primary_gpu_bytes
            )
            / layer_double["runtime"]["weight_gpu_bytes"]
        ),
        "matrix_double_vs_matrix_serial_speedup": (
            decode_median(matrix_serial) / decode_median(primary)
        ),
        "layer_double_vs_layer_serial_speedup": (
            decode_median(layer_serial) / decode_median(layer_double)
        ),
        "primary_vs_layer_serial_speedup": (
            decode_median(layer_serial) / decode_median(primary)
        ),
        "primary_vs_layer_double_latency_delta_pct": (
            100.0
            * (decode_median(primary) - decode_median(layer_double))
            / decode_median(layer_double)
        ),
        "full_pinned_vs_staging_speedup_matrix_s2": (
            decode_median(staging_matrix) / decode_median(primary)
        ),
        "gpu0_vs_gpu1_primary_latency_delta_pct": (
            100.0
            * abs(decode_median(primary) - decode_median(gpu1_primary))
            / decode_median(gpu1_primary)
        ),
    }

    shard_files = []
    for path in sorted(CHECKPOINT.glob("model-*.safetensors")):
        shard_files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
            }
        )
    shard_paths = sorted(CHECKPOINT.glob("model-*.safetensors"))
    tensor_inventory = inspect_safetensors(shard_paths)
    checkpoint = {
        "source": (
            "https://www.modelscope.cn/LLM-Research/"
            "Meta-Llama-3.1-8B.git"
        ),
        "revision": git_output("rev-parse", "HEAD"),
        "shards": shard_files,
        "safetensor_file_bytes": sum(
            item["size_bytes"] for item in shard_files
        ),
        **tensor_inventory,
        "optional_lfs_assets_not_required_for_transformers": [
            "original/consolidated.00.pth",
            "original/tokenizer.model",
        ],
    }

    checks = []

    def check(name, condition, detail):
        checks.append(
            {
                "name": name,
                "passed": bool(condition),
                "detail": detail,
            }
        )

    check(
        "checkpoint_safetensor_integrity",
        checkpoint["tensor_count"] == 291
        and checkpoint["parameter_elements"] == 8_030_261_248
        and checkpoint["parameter_payload_bytes"]
        == parameter_payload_bytes
        and checkpoint["tensor_dtype_counts"] == {"BF16": 291},
        (
            f"{checkpoint['tensor_count']} BF16 tensors, "
            f"{checkpoint['parameter_elements']:,} elements, "
            f"{checkpoint['parameter_payload_bytes']:,} payload bytes"
        ),
    )
    check(
        "checkpoint_shard_count",
        len(shard_files) == 4,
        f"found {len(shard_files)} safetensor shards",
    )
    check(
        "all_clean_repeats_present",
        all(row["decode_repeats"] == 7 for row in rows),
        "all eight configurations contain seven clean decode samples",
    )
    check(
        "no_hot_path_disk_reads",
        all(
            row["measured_decode_disk_read_bytes"] == 0
            for row in rows
        ),
        "process I/O read_bytes delta is zero in every decode window",
    )
    check(
        "correctness_token_match",
        correctness["generated_tokens_exact_match"]
        and all(correctness["argmax_match_per_step"]),
        "8/8 greedy token argmax decisions match Transformers reference",
    )
    check(
        "correctness_cosine",
        min(correctness["cosine_similarity_per_step"]) > 0.9997,
        (
            "minimum full-vocabulary logit cosine="
            f"{min(correctness['cosine_similarity_per_step']):.8f}"
        ),
    )
    check(
        "full_pinned_semantics",
        primary["cpu_arena_is_pinned"]
        and primary["cpu_pinned_bytes"] == parameter_payload_bytes,
        "PyTorch reports the complete weight arena as pinned",
    )
    check(
        "staging_semantics",
        not staging_matrix["cpu_arena_is_pinned"]
        and all(staging_matrix["staging_slots_are_pinned"])
        and staging_matrix["cpu_pinned_bytes"]
        == staging_matrix["runtime"]["device_slots_bytes"],
        "pageable arena plus two pinned staging slots",
    )
    check(
        "cross_gpu_reproducibility",
        comparisons["gpu0_vs_gpu1_primary_latency_delta_pct"] < 1.0,
        (
            "primary median latency differs by "
            f"{comparisons['gpu0_vs_gpu1_primary_latency_delta_pct']:.3f}%"
        ),
    )
    check(
        "stream_byte_consistency",
        all(
            profile["h2d_bytes"] == 13_958_643_712
            for report in loaded.values()
            for profile in report["profiles"]
        ),
        "every profile transferred the same 13,958,643,712 bytes/token",
    )

    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "On a single-user RTX 3080 Ti system, how much VRAM does "
            "real Llama-3.1-8B CPU-resident weight streaming save, and "
            "how do pinning, granularity, and one/two slots affect decode?"
        ),
        "scope": {
            "model": "Meta-Llama-3.1-8B BF16",
            "prompt": "The meaning of life is",
            "prompt_tokens": 6,
            "batch_size": 1,
            "decode_metric": "wall-clock milliseconds per one-token decode",
            "decode_samples_per_configuration": 7,
            "warmup_decode_steps": 1,
            "profile_samples_per_configuration": 3,
            "storage_hierarchy": "CPU DRAM to GPU; SSD excluded from hot path",
            "gpu_model": "NVIDIA GeForce RTX 3080 Ti",
            "gpu_capacity_bytes": gpu_capacity_bytes,
        },
        "checkpoint": checkpoint,
        "correctness": correctness,
        "benchmark_rows": rows,
        "comparisons": comparisons,
        "qa": {
            "passed": all(item["passed"] for item in checks),
            "passed_count": sum(item["passed"] for item in checks),
            "total_count": len(checks),
            "checks": checks,
        },
        "sources": sources,
        "limitations": [
            (
                "The 14.958 GiB BF16 parameter payload exceeds the "
                "11.668 GiB physical GPU, so an ordinary full-GPU "
                "Llama-3.1-8B speed baseline cannot run on this machine."
            ),
            (
                "Latency results are batch=1 greedy decode for one short "
                "prompt; prompt length, batching, quantization, and other "
                "GPU/PCIe platforms may change the balance."
            ),
            (
                "CUDA event component sums are diagnostic. Events from "
                "overlapping streams and profiling callbacks are not "
                "additive wall-clock decompositions."
            ),
            (
                "Linux VmLck does not account CUDA page-locked allocations "
                "on this driver; pinned status is verified through "
                "Tensor.is_pinned() and allocation sizes."
            ),
        ],
    }
    output_path = RESULTS / "real_llama31_summary.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if not output["qa"]["passed"]:
        raise SystemExit("one or more validation checks failed")


if __name__ == "__main__":
    main()
