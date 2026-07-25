#!/usr/bin/env python3
"""Build the canonical portable-report artifact from saved benchmark JSON."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


MIB = 1024 * 1024
LAYER_BYTES = 121_643_008
MODEL_TENSOR_BYTES = 2_471_628_800
DECODER_BYTES = 1_946_288_128
H2D_BYTES = (
    65_536,
    1_048_576,
    2_097_152,
    8_388_608,
    20_971_520,
    33_554_432,
    100_663_296,
    121_643_008,
    268_435_456,
)
COMPUTE_ROWS = (1, 128, 512, 1024, 2048, 4096)


def load(path):
    return json.loads(path.read_text())


def async_h2d_row(data, nbytes):
    return next(
        row
        for row in data["transfer_results"]
        if row["memory"] == "pinned"
        and row["api"] == "cuMemcpyHtoDAsync_v2"
        and row["size_bytes"] == nbytes
    )


def source_manifest(source_id, label, path=None, href=None):
    result = {"id": source_id, "label": label}
    if path:
        result["path"] = path
    if href:
        result["href"] = href
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifact.json"))
    parser.add_argument(
        "--notes-output", type=Path, default=Path("report_notes.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    generated_at = datetime.now(timezone.utc).isoformat()

    raw_h2d = {
        gpu: load(root / "h2d_results_gpu{}.json".format(gpu))
        for gpu in (0, 1)
    }
    torch_results = {
        gpu: load(root / "results/torch_llama32_1b_gpu{}.json".format(gpu))
        for gpu in (0, 1)
    }
    arena = load(root / "results/pinned_arena_probe.json")
    validation = load(root / "results/validation.json")

    exact_h2d = {
        gpu: async_h2d_row(raw_h2d[gpu], LAYER_BYTES) for gpu in (0, 1)
    }
    raw_bw_gpu0 = exact_h2d[0]["gpu_effective_gbps"]["median"]
    decoder_h2d_ms = DECODER_BYTES / (raw_bw_gpu0 * 1e9) * 1000
    decoder_h2d_tps = 1000 / decoder_h2d_ms

    pipeline_curve = []
    pipeline_chart = []
    for gpu0_row, gpu1_row in zip(
        torch_results[0]["pipeline"]["rows"],
        torch_results[1]["pipeline"]["rows"],
    ):
        if gpu0_row["rows_M"] != gpu1_row["rows_M"]:
            raise ValueError("GPU pipeline row mismatch")
        curve_row = {
            "M": gpu0_row["rows_M"],
            "M_label": str(gpu0_row["rows_M"]),
            "gpu0_speedup": gpu0_row["speedup_vs_sequential"],
            "gpu1_speedup": gpu1_row["speedup_vs_sequential"],
            "gpu0_sequential_ms": gpu0_row["modes"]["sequential"][
                "total_ms"
            ]["median"],
            "gpu0_overlap_ms": gpu0_row["modes"]["overlap"]["total_ms"][
                "median"
            ],
            "gpu1_sequential_ms": gpu1_row["modes"]["sequential"][
                "total_ms"
            ]["median"],
            "gpu1_overlap_ms": gpu1_row["modes"]["overlap"]["total_ms"][
                "median"
            ],
        }
        pipeline_curve.append(curve_row)
        for gpu in (0, 1):
            pipeline_chart.append(
                {
                    "M": curve_row["M"],
                    "M_label": curve_row["M_label"],
                    "gpu": "GPU{}".format(gpu),
                    "speedup": curve_row["gpu{}_speedup".format(gpu)],
                    "sequential_ms": curve_row[
                        "gpu{}_sequential_ms".format(gpu)
                    ],
                    "overlap_ms": curve_row[
                        "gpu{}_overlap_ms".format(gpu)
                    ],
                }
            )

    h2d_chunks = []
    for nbytes in H2D_BYTES:
        rows = [async_h2d_row(raw_h2d[gpu], nbytes) for gpu in (0, 1)]
        h2d_chunks.append(
            {
                "size_bytes": nbytes,
                "size": (
                    "{:.3f} MiB".format(nbytes / MIB)
                    if nbytes != LAYER_BYTES
                    else "116.008 MiB"
                ),
                "gpu0_event_us": rows[0]["gpu_event_us"]["median"],
                "gpu0_GBps": rows[0]["gpu_effective_gbps"]["median"],
                "gpu1_event_us": rows[1]["gpu_event_us"]["median"],
                "gpu1_GBps": rows[1]["gpu_effective_gbps"]["median"],
            }
        )

    compute_gpu0 = []
    compute_lookup = {
        row["rows_M"]: row for row in torch_results[0]["compute"]
    }
    for m in COMPUTE_ROWS:
        row = compute_lookup[m]
        compute_gpu0.append(
            {
                "M": m,
                "elapsed_ms": row["gpu_elapsed_ms"]["median"],
                "effective_TFLOPs": row["effective_TFLOPs"],
                "nominal_weight_GBps": row[
                    "nominal_weight_bytes_per_elapsed_GBps"
                ],
            }
        )

    granularity = []
    recommendations = {
        1: "2/8 MiB tensor；只能改善首片延迟，总体仍 H2D-bound",
        512: "32 MiB MLP matrix",
        2048: "96 MiB MLP 或 116 MiB 整层",
        4096: "116 MiB 整层",
    }
    for m in (1, 512, 2048, 4096):
        elapsed_ms = compute_lookup[m]["gpu_elapsed_ms"]["median"]
        granularity.append(
            {
                "M": m,
                "compute_ms": elapsed_ms,
                "hideable_MiB_at_24_GBps": (
                    elapsed_ms / 1000 * 24e9 / MIB
                ),
                "recommendation": recommendations[m],
            }
        )

    model_layout = [
        {
            "component": "tied embedding / lm_head",
            "bytes": 525_336_576,
            "MiB": 501.0,
        },
        {
            "component": "single decoder layer + norms",
            "bytes": LAYER_BYTES,
            "MiB": LAYER_BYTES / MIB,
        },
        {
            "component": "16 decoder layers",
            "bytes": DECODER_BYTES,
            "MiB": DECODER_BYTES / MIB,
        },
        {"component": "final norm", "bytes": 4096, "MiB": 4096 / MIB},
        {
            "component": "all tensor payload",
            "bytes": MODEL_TENSOR_BYTES,
            "MiB": MODEL_TENSOR_BYTES / MIB,
        },
    ]

    best_row = max(
        pipeline_curve,
        key=lambda row: max(row["gpu0_speedup"], row["gpu1_speedup"]),
    )
    headline_h2d = [
        {
            "raw_GBps": raw_bw_gpu0,
            "exact_layer_ms": exact_h2d[0]["gpu_event_us"]["median"] / 1000,
            "enqueue_us": exact_h2d[0]["host_call_us"]["median"],
            "decoder_h2d_tps_lower_bound": decoder_h2d_tps,
        }
    ]
    headline_pipeline = [
        {
            "best_speedup": max(
                best_row["gpu0_speedup"], best_row["gpu1_speedup"]
            ),
            "best_M": best_row["M"],
        }
    ]
    headline_arena = [
        {
            "model_GiB": MODEL_TENSOR_BYTES / (1024 ** 3),
            "allocation_s": arena["allocation_seconds"],
            "first_touch_GBps": arena["first_touch_GBps"],
        }
    ]

    manifest_sources = [
        source_manifest(
            "pipeline_chart_sql",
            "Reviewed two-GPU pipeline chart selection",
            path="queries/pipeline_chart.sql",
        ),
        source_manifest(
            "pipeline_table_sql",
            "Reviewed two-GPU pipeline table selection",
            path="queries/pipeline_table.sql",
        ),
        source_manifest(
            "h2d_table_sql",
            "Reviewed CUDA Driver H2D table selection",
            path="queries/h2d_table.sql",
        ),
        source_manifest(
            "compute_table_sql",
            "Reviewed GPU0 projection table selection",
            path="queries/compute_gpu0.sql",
        ),
        source_manifest(
            "model_layout_sql",
            "Reviewed model-layout table selection",
            path="queries/model_layout.sql",
        ),
        source_manifest(
            "granularity_sql",
            "Reviewed prefetch-granularity table selection",
            path="queries/granularity.sql",
        ),
        source_manifest(
            "h2d_driver",
            "CUDA Driver API H2D raw results",
            path="h2d_results_summary.json",
        ),
        source_manifest(
            "torch_pipeline",
            "Two-GPU projection and pipeline benchmark",
            path="benchmarks/build_report_artifact.py",
        ),
        source_manifest(
            "torch_compute_gpu0",
            "GPU0 exact-shape projection benchmark",
            path="results/torch_llama32_1b_gpu0.json",
        ),
        source_manifest(
            "pinned_arena",
            "Full-model pinned CPU arena probe",
            path="results/pinned_arena_probe.json",
        ),
        source_manifest(
            "validation",
            "Benchmark QA receipt",
            path="results/validation.json",
        ),
        source_manifest(
            "model_card",
            "Official Llama-3.2-1B model card",
            href="https://huggingface.co/meta-llama/Llama-3.2-1B",
        ),
        source_manifest(
            "airllm_source",
            "AirLLM audited source revision",
            href=(
                "https://github.com/lyogavin/airllm/tree/"
                "17677cb821016b36a0610c8e1f2befab030d1942"
            ),
        ),
        source_manifest(
            "pytorch_overlap_docs",
            "PyTorch pinned-memory and non-blocking copy guide",
            href=(
                "https://docs.pytorch.org/tutorials/intermediate/"
                "pinmem_nonblock.html"
            ),
        ),
    ]

    sources = [
        {
            "id": "pipeline_chart_sql",
            "label": "Reviewed two-GPU pipeline chart selection",
            "path": "queries/pipeline_chart.sql",
            "query": {
                "engine": "SQLite-compatible snapshot selection",
                "language": "sql",
                "sql": (
                    "SELECT M, M_label, gpu, speedup, sequential_ms, "
                    "overlap_ms FROM pipeline_chart ORDER BY M, gpu"
                ),
                "description": (
                    "Selects all 16 reviewed GPU/M observations used by the "
                    "native report chart."
                ),
                "executed_at": generated_at,
                "tables_used": ["pipeline_chart"],
                "filters": ["No interpolation; all eight M points per GPU"],
                "metric_definitions": [
                    "speedup = matched sequential total ms / overlap total ms"
                ],
            },
        },
        {
            "id": "pipeline_table_sql",
            "label": "Reviewed two-GPU pipeline table selection",
            "path": "queries/pipeline_table.sql",
            "query": {
                "engine": "SQLite-compatible snapshot selection",
                "language": "sql",
                "sql": (
                    "SELECT M, gpu0_sequential_ms, gpu0_overlap_ms, "
                    "gpu0_speedup, gpu1_sequential_ms, gpu1_overlap_ms, "
                    "gpu1_speedup FROM pipeline_curve ORDER BY M"
                ),
                "description": (
                    "Selects the eight matched wide rows used for exact lookup."
                ),
                "executed_at": generated_at,
                "tables_used": ["pipeline_curve"],
            },
        },
        {
            "id": "h2d_table_sql",
            "label": "Reviewed CUDA Driver H2D table selection",
            "path": "queries/h2d_table.sql",
            "query": {
                "engine": "SQLite-compatible snapshot selection",
                "language": "sql",
                "sql": (
                    "SELECT size_bytes, size, gpu0_event_us, gpu0_GBps, "
                    "gpu1_event_us, gpu1_GBps FROM h2d_chunks "
                    "ORDER BY size_bytes"
                ),
                "description": (
                    "Selects reconciled pinned async medians at natural "
                    "Llama tensor boundaries."
                ),
                "executed_at": generated_at,
                "tables_used": ["h2d_chunks"],
            },
        },
        {
            "id": "compute_table_sql",
            "label": "Reviewed GPU0 projection table selection",
            "path": "queries/compute_gpu0.sql",
            "query": {
                "engine": "SQLite-compatible snapshot selection",
                "language": "sql",
                "sql": (
                    "SELECT M, elapsed_ms, effective_TFLOPs, "
                    "nominal_weight_GBps FROM compute_gpu0 ORDER BY M"
                ),
                "description": (
                    "Selects the reviewed exact-shape projection calibration."
                ),
                "executed_at": generated_at,
                "tables_used": ["compute_gpu0"],
            },
        },
        {
            "id": "model_layout_sql",
            "label": "Reviewed model-layout table selection",
            "path": "queries/model_layout.sql",
            "query": {
                "engine": "SQLite-compatible snapshot selection",
                "language": "sql",
                "sql": (
                    "SELECT component, bytes, MiB FROM model_layout "
                    "ORDER BY bytes DESC"
                ),
                "description": (
                    "Selects the reviewed BF16 tensor-layout components."
                ),
                "executed_at": generated_at,
                "tables_used": ["model_layout"],
            },
        },
        {
            "id": "granularity_sql",
            "label": "Reviewed prefetch-granularity table selection",
            "path": "queries/granularity.sql",
            "query": {
                "engine": "SQLite-compatible snapshot selection",
                "language": "sql",
                "sql": (
                    "SELECT M, compute_ms, hideable_MiB_at_24_GBps, "
                    "recommendation FROM granularity ORDER BY M"
                ),
                "description": (
                    "Selects the 24 GB/s compute-window calculation."
                ),
                "executed_at": generated_at,
                "tables_used": ["granularity"],
                "metric_definitions": [
                    "hideable MiB = compute seconds × 24e9 / 2^20"
                ],
            },
        },
        {
            "id": "h2d_driver",
            "label": "CUDA Driver API H2D raw results",
            "path": "h2d_results_summary.json",
            "query": {
                "engine": "CUDA Driver API",
                "language": "C11",
                "description": (
                    "Pinned/pageable H2D CUDA Event medians and batched "
                    "tiny-copy enqueue fits on both RTX 3080 Ti GPUs."
                ),
                "executed_at": raw_h2d[0]["timestamp_utc"],
                "tables_used": [
                    "h2d_results_gpu0.json",
                    "h2d_results_gpu1.json",
                ],
                "metric_definitions": [
                    "GB/s = transferred bytes / CUDA Event elapsed seconds / 1e9",
                    "Host enqueue is wall time around one cuMemcpyHtoDAsync_v2 call",
                ],
            },
        },
        {
            "id": "torch_pipeline",
            "label": "Two-GPU projection and pipeline benchmark",
            "path": "benchmarks/build_report_artifact.py",
            "query": {
                "engine": "Python",
                "language": "python",
                "description": (
                    "Joins matching M rows from the two saved PyTorch JSON "
                    "files without interpolation."
                ),
                "executed_at": generated_at,
                "tables_used": [
                    "results/torch_llama32_1b_gpu0.json",
                    "results/torch_llama32_1b_gpu1.json",
                ],
                "metric_definitions": [
                    "M = batch × tokens processed by one decoder-layer call",
                    "Pipeline speedup = matched sequential total ms / overlap total ms",
                    "Both modes use 16 stages, two preallocated device slots, and persistent streams",
                ],
            },
        },
        {
            "id": "torch_compute_gpu0",
            "label": "GPU0 exact-shape projection benchmark",
            "path": "results/torch_llama32_1b_gpu0.json",
            "query": {
                "engine": "PyTorch CUDA Event",
                "language": "python",
                "description": (
                    "Exact Llama-3.2-1B q/k/v/o and gate/up/down BF16 "
                    "projection shapes on GPU0."
                ),
                "executed_at": torch_results[0]["environment"][
                    "timestamp_utc"
                ],
                "metric_definitions": [
                    "Projection FLOPs = 2 × linear weight elements × M",
                    "The benchmark omits attention, RMSNorm, RoPE, KV-cache, and residual work",
                ],
            },
        },
        {
            "id": "pinned_arena",
            "label": "Full-model pinned CPU arena probe",
            "path": "results/pinned_arena_probe.json",
            "query": {
                "engine": "CUDA Driver API",
                "language": "python",
                "description": (
                    "cuMemHostAlloc, libc memset first-touch, and "
                    "cuMemFreeHost for the exact model tensor payload."
                ),
                "executed_at": arena["timestamp_utc"],
            },
        },
        {
            "id": "validation",
            "label": "Benchmark QA receipt",
            "path": "results/validation.json",
            "query": {
                "engine": "Python",
                "language": "python",
                "description": (
                    "Reconciles raw hashes, exact-layer H2D medians, formulas, "
                    "sample rows, hidden-fraction scope, and two-GPU agreement."
                ),
                "executed_at": validation["generated_at"],
            },
        },
        {
            "id": "model_card",
            "label": "Official Llama-3.2-1B model card",
            "href": "https://huggingface.co/meta-llama/Llama-3.2-1B",
        },
        {
            "id": "airllm_source",
            "label": "AirLLM audited source revision",
            "href": (
                "https://github.com/lyogavin/airllm/tree/"
                "17677cb821016b36a0610c8e1f2befab030d1942"
            ),
        },
        {
            "id": "pytorch_overlap_docs",
            "label": "PyTorch pinned-memory and non-blocking copy guide",
            "href": (
                "https://docs.pytorch.org/tutorials/intermediate/"
                "pinmem_nonblock.html"
            ),
        },
    ]

    title = "Llama-3.2-1B CPU 常驻与 GPU 异步分层推理基线"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "双 RTX 3080 Ti 上的 H2D、精确 projection 计算、"
                "双 slot 流水与 AirLLM 调度改进建议。"
            ),
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "pipeline_speedup_chart",
                    "title": "16-stage 双 slot 流水加速比",
                    "subtitle": (
                        "speedup = matched sequential / overlap；"
                        "BF16 exact-shape projection-only，2026-07-25"
                    ),
                    "intent": "trend",
                    "question": "流水收益如何随 M=batch×tokens 改变，两张卡是否一致？",
                    "rationale": (
                        "M 是有语义顺序的离散校准点；双线同时展示收益形状"
                        "与两卡复现性，比只列峰值更能揭示 copy/compute 交点。"
                    ),
                    "comparisonContext": {
                        "baseline": "matched sequential copy-then-compute",
                        "grain": "M=batch×tokens; 16 decoder stages",
                        "unit": "speedup ×",
                    },
                    "type": "line",
                    "dataset": "pipeline_chart",
                    "sourceId": "pipeline_chart_sql",
                    "encodings": {
                        "x": {
                            "field": "M_label",
                            "type": "ordinal",
                            "label": "M = batch × tokens",
                        },
                        "y": {
                            "field": "speedup",
                            "type": "quantitative",
                            "label": "加速比",
                            "unit": "×",
                        },
                        "color": {
                            "field": "gpu",
                            "type": "nominal",
                            "label": "GPU",
                        },
                        "lineStyle": {
                            "field": "gpu",
                            "type": "nominal",
                            "label": "GPU",
                        },
                        "tooltip": [
                            {
                                "field": "M",
                                "type": "quantitative",
                                "label": "M",
                            },
                            {
                                "field": "gpu",
                                "type": "nominal",
                                "label": "GPU",
                            },
                            {
                                "field": "sequential_ms",
                                "type": "quantitative",
                                "label": "Serial ms",
                            },
                            {
                                "field": "overlap_ms",
                                "type": "quantitative",
                                "label": "Overlap ms",
                            },
                        ],
                    },
                    "valueFormat": "number",
                    "unit": "×",
                    "layout": "full",
                    "palette": {
                        "kind": "categorical",
                        "name": "two-GPU comparison",
                    },
                    "legend": {"position": "bottom", "sort": "spec"},
                    "labels": {"values": "endpoints"},
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": 1.0,
                            "label": "无加速",
                            "color": "neutral",
                            "lineStyle": "dotted",
                        }
                    ],
                    "surface": {
                        "surface": "card",
                        "viewMode": "both",
                        "interactiveLegend": True,
                    },
                }
            ],
            "tables": [
                {
                    "id": "pipeline_table",
                    "title": "两张卡的 16-stage 流水结果",
                    "subtitle": "同一 M、同一双 slot/event 图；时间为各 run total median。",
                    "dataset": "pipeline_curve",
                    "sourceId": "pipeline_table_sql",
                    "defaultSort": {"field": "M", "direction": "asc"},
                    "density": "spacious",
                    "columns": [
                        {"field": "M", "label": "M", "format": "number"},
                        {
                            "field": "gpu0_sequential_ms",
                            "label": "GPU0 serial ms",
                            "format": "number",
                        },
                        {
                            "field": "gpu0_overlap_ms",
                            "label": "GPU0 overlap ms",
                            "format": "number",
                        },
                        {
                            "field": "gpu0_speedup",
                            "label": "GPU0 speedup ×",
                            "format": "number",
                        },
                        {
                            "field": "gpu1_speedup",
                            "label": "GPU1 speedup ×",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "h2d_table",
                    "title": "Pinned H2D 的自然权重分片",
                    "subtitle": "CUDA Driver API median；GB/s 为十进制单位。",
                    "dataset": "h2d_chunks",
                    "sourceId": "h2d_table_sql",
                    "defaultSort": {"field": "size_bytes", "direction": "asc"},
                    "density": "spacious",
                    "columns": [
                        {
                            "field": "size_bytes",
                            "label": "Bytes",
                            "format": "number",
                        },
                        {"field": "size", "label": "分片", "type": "text"},
                        {
                            "field": "gpu0_event_us",
                            "label": "GPU0 event µs",
                            "format": "number",
                        },
                        {
                            "field": "gpu0_GBps",
                            "label": "GPU0 GB/s",
                            "format": "number",
                        },
                        {
                            "field": "gpu1_GBps",
                            "label": "GPU1 GB/s",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "compute_table",
                    "title": "GPU0 单层 projection 计算",
                    "subtitle": "精确 q/k/v/o 与 gate/up/down 形状；不含完整 attention。",
                    "dataset": "compute_gpu0",
                    "sourceId": "compute_table_sql",
                    "defaultSort": {"field": "M", "direction": "asc"},
                    "density": "spacious",
                    "columns": [
                        {"field": "M", "label": "M", "format": "number"},
                        {
                            "field": "elapsed_ms",
                            "label": "计算 ms",
                            "format": "number",
                        },
                        {
                            "field": "effective_TFLOPs",
                            "label": "有效 TFLOP/s",
                            "format": "number",
                        },
                        {
                            "field": "nominal_weight_GBps",
                            "label": "Nominal weight/time GB/s",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "model_layout_table",
                    "title": "Llama-3.2-1B BF16 tensor 布局",
                    "subtitle": "Hub 元数据 revision 4e20de3；tied embedding/head 只计一次。",
                    "dataset": "model_layout",
                    "sourceId": "model_layout_sql",
                    "defaultSort": {"field": "bytes", "direction": "desc"},
                    "density": "spacious",
                    "columns": [
                        {
                            "field": "component",
                            "label": "组成",
                            "type": "text",
                        },
                        {"field": "bytes", "label": "Bytes", "format": "number"},
                        {"field": "MiB", "label": "MiB", "format": "number"},
                    ],
                },
                {
                    "id": "granularity_table",
                    "title": "按计算窗口选择预取粒度",
                    "subtitle": "可隐藏字节按 24 GB/s × GPU0 projection 时间估算。",
                    "dataset": "granularity",
                    "sourceId": "granularity_sql",
                    "defaultSort": {"field": "M", "direction": "asc"},
                    "density": "spacious",
                    "columns": [
                        {"field": "M", "label": "M", "format": "number"},
                        {
                            "field": "compute_ms",
                            "label": "整层计算 ms",
                            "format": "number",
                        },
                        {
                            "field": "hideable_MiB_at_24_GBps",
                            "label": "窗口可隐藏 MiB",
                            "format": "number",
                        },
                        {
                            "field": "recommendation",
                            "label": "建议边界",
                            "type": "text",
                        },
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# " + title},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## 技术摘要\n\n"
                        "- **硬件路径可行。** 两张 RTX 3080 Ti 都有独立 DMA engine；"
                        "精确 116.008 MiB layer 的 raw pinned H2D 约 5.0 ms。\n"
                        "- **收益取决于 M。** M=1 的 16-stage 加速只有约 1.03×，"
                        "M=2048 达约 1.71×；约 M=2.5k–2.7k 后整层 copy 可被计算覆盖。\n"
                        "- **1B 首版应全量 pinned CPU 常驻。** 2.302 GiB arena 已成功分配；"
                        "pageable H2D 会 host 阻塞，单线程 pageable→pinned staging 也更慢。\n"
                        "- **结论是校准级、非端到端。** 官方权重受 manual gate 限制，"
                        "本轮计算是精确形状 projection-only；真实 logits/token/s 待授权后验证。"
                    ),
                },
                {
                    "id": "pipeline_finding",
                    "type": "markdown",
                    "sourceId": "torch_pipeline",
                    "body": (
                        "## 重叠在 copy/compute 接近平衡时最有效\n\n"
                        "两张卡的曲线几乎重合：M=1 只有约 1.03×，M=512/1024 "
                        "约 1.20×/1.37×，M=2048 约 1.71×。M=4096 时计算已经"
                        "长于 H2D，流水仍完全隐藏 copy，但总时间转由 compute 决定，"
                        "因此相对加速回落到约 1.59×。这说明粒度优化可以去掉气泡，"
                        "但不能减少每 token 经过 PCIe 的总权重字节。"
                    ),
                },
                {
                    "id": "pipeline_chart_block",
                    "type": "chart",
                    "chartId": "pipeline_speedup_chart",
                },
                {
                    "id": "pipeline_table_context",
                    "type": "markdown",
                    "sourceId": "torch_pipeline",
                    "body": (
                        "### 精确时间验证曲线不是计时假象\n\n"
                        "串行与重叠模式复用同一组 persistent stream/event 和两个真实"
                        " device slot；copy 写入的 slot 就是随后 compute 读取的权重。"
                        "表中保留绝对时间，便于检查加速比与填充/排空开销。"
                    ),
                },
                {
                    "id": "pipeline_table_block",
                    "type": "table",
                    "tableId": "pipeline_table",
                },
                {
                    "id": "h2d_finding",
                    "type": "markdown",
                    "sourceId": "h2d_driver",
                    "body": (
                        "## 1–2 MiB 已接近满带宽，pageable 路径不能重叠\n\n"
                        "GPU0/GPU1 的 116.008 MiB raw H2D 分别为 5.020/5.055 ms，"
                        "约 24.23/24.07 GB/s；单次 host enqueue 约 1.8 µs。"
                        "64 KiB 只有约 12 GB/s，而 1–2 MiB 已达到约 24.6–25.3 GB/s。"
                        "因此最小调度片应为 1–2 MiB，并优先使用 2/8/32 MiB 等自然矩阵边界。"
                    ),
                },
                {"id": "h2d_table_block", "type": "table", "tableId": "h2d_table"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": (
                        "## 范围、数据与指标定义\n\n"
                        "**M** 是一次 decoder-layer 调用中的 `batch × tokens`。"
                        "**Pipeline speedup** 是完全相同 16-stage 工作在 matched "
                        "sequential 与 overlap 模式下的总时长之比。H2D GB/s 使用十进制"
                        "字节；MiB/GiB 使用二进制单位。模型 tensor payload 为 "
                        "2,471,628,800 B；单层含两个 norm 为 121,643,008 B。"
                    ),
                },
                {
                    "id": "model_layout_context",
                    "type": "markdown",
                    "sourceId": "model_card",
                    "body": (
                        "### 1B 是可控的强制 out-of-core 实验\n\n"
                        "模型本身能完整放入 12 GiB GPU，因此可以在同卡建立"
                        " full-resident reference，再强制启用 CPU streaming，直接比较"
                        "每层输出、最终 logits、TTFT、token/s 与显存峰值。"
                    ),
                },
                {
                    "id": "model_layout_block",
                    "type": "table",
                    "tableId": "model_layout_table",
                },
                {
                    "id": "compute_finding",
                    "type": "markdown",
                    "sourceId": "torch_compute_gpu0",
                    "body": (
                        "## Projection 计算约在 M≈2.6k 与整层 H2D 相交\n\n"
                        "GPU0 单层 projection 在 M=1/512/2048/4096 时约为 "
                        "0.212/1.286/4.035/7.733 ms。与约 5.0 ms 的整层 raw H2D"
                        "比较，decode 小 M 明显 copy-bound；大 prefill 才适合整层双缓冲。"
                    ),
                },
                {
                    "id": "compute_table_block",
                    "type": "table",
                    "tableId": "compute_table",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "validation",
                    "body": (
                        "## 方法与稳健性检查\n\n"
                        "低层 H2D 直接 `dlopen(libcuda)`，由 CUDA Event 包围 DMA；"
                        "host latency 只包围 API 调用。PyTorch 流水使用固定 stream/event、"
                        "两个预分配 slot，完整 pipeline 末尾才同步。QA 脚本重算 SHA-256、"
                        "exact-layer median、speedup 与 hidden-fraction 公式，并检查两卡一致性；"
                        "65/65 项通过。当前结论评级为 **share with caveats**。"
                    ),
                },
                {
                    "id": "airllm_gap",
                    "type": "markdown",
                    "sourceId": "airllm_source",
                    "body": (
                        "## AirLLM 的缺口在 H2D 与 allocator 热路径\n\n"
                        "审查 revision `17677cb` 后确认：worker 预取只覆盖 disk→CPU；"
                        "pre-hook 仍同步逐 tensor 搬 GPU，且下一层 CPU prefetch 在当前"
                        " H2D 后才提交。`pin_memory()` 返回值未写回，post-hook 每层还执行"
                        " `to(meta)`、GC、`malloc_trim` 与 `empty_cache`。首版更适合保留"
                        "checkpoint/模型兼容思路，重写独立 runtime scheduler。"
                    ),
                },
                {
                    "id": "scheduler_design",
                    "type": "markdown",
                    "sourceId": "torch_pipeline",
                    "body": (
                        "## 首版采用 pinned arena + 双 slot + ready/free event\n\n"
                        "启动时解析 safetensors header，将全量 BF16 tensor 直接读入连续"
                        " pinned arena；GPU 常驻 tied embedding/head、norm、RoPE 与 KV cache。"
                        "copy stream 写 slot B，记录 `ready[i]`；compute stream 等待 ready、"
                        "使用 slot 后记录 `free[i]`；覆盖 slot 前只等待对应 free。热路径不做"
                        " `module.to()`、分配/释放、`empty_cache()` 或全局同步。"
                    ),
                },
                {
                    "id": "granularity_context",
                    "type": "markdown",
                    "body": (
                        "### Autotuner 按计算窗口选择下一片\n\n"
                        "用 `S_prefetch ≈ B_H2D × T_compute(window)` 选择当前计算窗口内"
                        "可隐藏的下一批字节，并贴合 K/V 2 MiB、Q/O 8 MiB、MLP matrix "
                        "32 MiB 等边界。小 M 下分片只改善首片延迟，无法改变 PCIe 总吞吐上限。"
                    ),
                },
                {
                    "id": "granularity_table_block",
                    "type": "table",
                    "tableId": "granularity_table",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 限制与不确定性\n\n"
                        "- 官方权重 manual-gated，本机无 token；未验证真实 logits 或生成速度。\n"
                        "- projection-only 不含 RMSNorm、RoPE、QK-softmax-V、KV-cache 与 residual。\n"
                        "- GPU clocks 未锁；matched sequential/overlap 比 tiny-M component slowdown 更可靠。\n"
                        "- timing-enabled dependency event 对两种模式影响相同，但生产绝对开销会更低。\n"
                        "- 全量 pinned 适合本机 1B；更大模型需评估 page-locked RAM 对系统的影响。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 推荐实施顺序\n\n"
                        "1. 本机接受模型 license，并通过环境变量或 `huggingface-cli login` "
                        "配置 token；checkpoint 放 `/ssd`。\n"
                        "2. 实现 safetensors offset manifest 与 2.302 GiB pinned arena loader。\n"
                        "3. 先完成整层双 slot runtime，并逐层对齐 full-resident reference。\n"
                        "4. 再实现 Q/K/V/O/MLP projection 粒度 schedule 与 autotuner。\n"
                        "5. 用 Nsight Systems 验证 copy engine/kernel overlap，并报告 TTFT、"
                        "token/s、峰值 VRAM 与 logits 误差。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 下一轮需要回答的问题\n\n"
                        "- 真实 Llama decoder 的 attention/KV 工作会把整层交点移动多少？\n"
                        "- batch-1 decode 是否应常驻更多未来层，而不是继续细切当前层？\n"
                        "- projection 内部 tiling 的 kernel/launch 成本是否抵消更早开算的收益？\n"
                        "- 扩展到大模型时，多线程 pageable→pinned staging 能否跑满本机 DRAM？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline_h2d": headline_h2d,
                "headline_pipeline": headline_pipeline,
                "headline_arena": headline_arena,
                "pipeline_curve": pipeline_curve,
                "pipeline_chart": pipeline_chart,
                "h2d_chunks": h2d_chunks,
                "compute_gpu0": compute_gpu0,
                "model_layout": model_layout,
                "granularity": granularity,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }

    output = args.output if args.output.is_absolute() else root / args.output
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    notes = {
        "generated_at": generated_at,
        "reporting_job": {
            "question": "Can CPU-resident Llama weights be streamed so H2D overlaps GPU compute on this machine?",
            "audience": "technical",
            "scope": "machine calibration and first-framework design",
            "comparison_baseline": "matched sequential copy-then-compute",
            "success_criteria": [
                "raw H2D and fixed overhead measured",
                "exact-shape projection compute measured",
                "true two-slot multi-stage overlap demonstrated",
                "limitations and next validation step explicit",
            ],
        },
        "required_structure_mapping": {
            "technical_summary": "technical_summary + headline_metrics",
            "key_findings_visual": "pipeline_finding + pipeline_speedup_chart",
            "scope_data_definitions": "scope_definitions + model_layout",
            "methodology": "methodology",
            "limitations_robustness": "limitations + validation source",
            "recommended_next_steps": "next_steps",
            "further_questions": "further_questions",
        },
        "chart_map": [
            {
                "section": "重叠在 copy/compute 接近平衡时最有效",
                "question": "How does 16-stage speedup change with M and reproduce across GPUs?",
                "family": "ordered-axis trend",
                "type": "line",
                "fields": [
                    "M_label",
                    "gpu",
                    "speedup",
                    "sequential_ms",
                    "overlap_ms",
                ],
                "claim": "Speedup peaks near M=2048 and both GPUs agree.",
                "palette": (
                    "hard two-root cap via categorical palette; GPU is also "
                    "encoded by line style"
                ),
                "delivery": "artifact.json -> report.html native chart",
            }
        ],
        "omission_notes": [
            "No H2D chart: exact lookup by natural tensor size is better served by the audit table.",
            "No end-to-end token/s chart: gated weights make that metric unavailable in this run.",
            "Markdown technical notes are supporting methodology, not a second report delivery surface.",
        ],
        "validation": {
            "assessment": validation["assessment"],
            "checks_passed": validation["checks_passed"],
            "checks_total": validation["checks_total"],
            "required_caveats": validation["required_caveats"],
        },
    }
    notes_output = (
        args.notes_output
        if args.notes_output.is_absolute()
        else root / args.notes_output
    )
    notes_output.write_text(
        json.dumps(notes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Wrote {}".format(output))
    print("Wrote {}".format(notes_output))


if __name__ == "__main__":
    main()
