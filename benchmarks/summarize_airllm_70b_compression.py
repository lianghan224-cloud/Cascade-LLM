#!/usr/bin/env python3
"""Compare CPU-resident AirLLM 8-bit compression with Cascade 70B."""

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "real_results" / "70b_int8"
AIRLLM_RESULTS = RESULTS / "airllm_compression"
GIB = 1024**3


def median_profile(report, key):
    return statistics.median(row[key] for row in report["profile_rows"])


def airllm_row(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    latency = statistics.median(report["decode_wall_ms"])
    return {
        "name": "airllm_8bit_cpu_{}".format(report["cpu_cache_mode"]),
        "implementation": "AirLLM compression=8bit",
        "cpu_cache_mode": report["cpu_cache_mode"],
        "median_token_ms": latency,
        "tokens_per_second": 1000.0 / latency,
        "compressed_h2d_bytes": median_profile(
            report, "compressed_h2d_bytes"
        ),
        "layer_load_decompress_ms": median_profile(
            report, "layer_load_decompress_wall_ms"
        ),
        "peak_gpu_allocated_bytes": report[
            "cuda_peak_allocated_bytes"
        ],
        "cpu_cache_bytes": report["cpu_cache_bytes"],
        "cpu_cache_pinned_bytes": report["cpu_cache_pinned_bytes"],
        "measured_decode_read_bytes": report[
            "measured_decode_io_delta"
        ]["read_bytes"],
        "generated_token_ids": report["generated_token_ids"],
        "generated_text": report["generated_text"],
        "source": str(path.relative_to(ROOT)),
    }


def cascade_rows():
    summary = json.loads(
        (RESULTS / "summary.json").read_text(encoding="utf-8")
    )
    selected = {
        "full_pinned_matrix_s2": "full_pinned",
        "pinned_staging_layer_s2": "pinned_staging",
    }
    rows = []
    for row in summary["benchmark_rows"]:
        if row["name"] not in selected:
            continue
        rows.append(
            {
                "name": "cascade_{}".format(selected[row["name"]]),
                "implementation": "Cascade",
                "cpu_cache_mode": selected[row["name"]],
                "median_token_ms": row["median_token_ms"],
                "tokens_per_second": row["tokens_per_second"],
                "compressed_h2d_bytes": summary["measurement"][
                    "weight_bytes_per_decode"
                ],
                "layer_load_decompress_ms": None,
                "peak_gpu_allocated_bytes": row[
                    "peak_gpu_allocated_bytes"
                ],
                "cpu_cache_bytes": summary["checkpoint_validation"][
                    "tensor_payload_bytes"
                ],
                "cpu_cache_pinned_bytes": row["pinned_cpu_bytes"],
                "measured_decode_read_bytes": None,
                "generated_token_ids": row["generated_token_ids"],
                "generated_text": summary["correctness"][
                    "generated_text"
                ],
                "source": "real_results/70b_int8/summary.json",
            }
        )
    return rows


def main():
    pageable_path = (
        AIRLLM_RESULTS / "bench_airllm_8bit_cpu_pageable.json"
    )
    pinned_path = AIRLLM_RESULTS / "bench_airllm_8bit_cpu_pinned.json"
    if not pageable_path.is_file():
        raise SystemExit("missing {}".format(pageable_path))

    airllm = [airllm_row(pageable_path)]
    if pinned_path.is_file():
        airllm.append(airllm_row(pinned_path))
    cascade = cascade_rows()
    by_name = {row["name"]: row for row in airllm + cascade}

    comparisons = {
        "cascade_staging_vs_airllm_pageable_speedup": (
            by_name["airllm_8bit_cpu_pageable"]["median_token_ms"]
            / by_name["cascade_pinned_staging"]["median_token_ms"]
        ),
        "cascade_full_pinned_vs_airllm_pageable_speedup": (
            by_name["airllm_8bit_cpu_pageable"]["median_token_ms"]
            / by_name["cascade_full_pinned"]["median_token_ms"]
        ),
    }
    if "airllm_8bit_cpu_pinned" in by_name:
        comparisons.update(
            {
                "airllm_pinned_vs_pageable_speedup": (
                    by_name["airllm_8bit_cpu_pageable"][
                        "median_token_ms"
                    ]
                    / by_name["airllm_8bit_cpu_pinned"][
                        "median_token_ms"
                    ]
                ),
                "cascade_full_pinned_vs_airllm_pinned_speedup": (
                    by_name["airllm_8bit_cpu_pinned"][
                        "median_token_ms"
                    ]
                    / by_name["cascade_full_pinned"][
                        "median_token_ms"
                    ]
                ),
            }
        )

    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": (
            "CPU-resident weights; SSD preparation and CPU cache preload "
            "excluded from decode latency"
        ),
        "rows": airllm + cascade,
        "comparisons": comparisons,
        "qa": {
            "airllm_pageable_decode_disk_read_is_negligible": (
                by_name["airllm_8bit_cpu_pageable"][
                    "measured_decode_read_bytes"
                ]
                < 16 * 1024**2
            ),
            "airllm_cpu_cache_is_materialized": (
                by_name["airllm_8bit_cpu_pageable"]["cpu_cache_bytes"]
                > 60 * GIB
            ),
        },
        "caveats": [
            (
                "AirLLM block-wise 8-bit compression and the RedHatAI "
                "per-channel W8A8 checkpoint are different quantizers."
            ),
            (
                "AirLLM decompresses each compressed module to floating "
                "point before Transformers computation; it does not use "
                "native INT8 activation GEMM in this mode."
            ),
        ],
    }
    output_path = AIRLLM_RESULTS / "summary.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if not all(output["qa"].values()):
        raise SystemExit("AirLLM comparison QA failed")


if __name__ == "__main__":
    main()
