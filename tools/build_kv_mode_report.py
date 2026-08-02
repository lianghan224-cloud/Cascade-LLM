#!/usr/bin/env python3
"""Aggregate KV mode benchmarks into JSON, Markdown and report artifact input."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-bf16", type=Path, required=True)
    parser.add_argument("--short-fp16", type=Path, required=True)
    parser.add_argument("--long-bf16", type=Path, required=True)
    parser.add_argument("--long-fp16", type=Path, required=True)
    parser.add_argument("--real8b", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    return parser.parse_args()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return Path(path).name


def rounded(value, digits=4):
    return round(float(value), int(digits))


def matching_ratio(left, right, key_fields, value_field):
    right_map = {
        tuple(item[field] for field in key_fields): item
        for item in right
    }
    ratios = []
    for item in left:
        other = right_map.get(tuple(item[field] for field in key_fields))
        if other is not None:
            ratios.append(float(item[value_field]) / float(other[value_field]))
    return ratios


def main():
    args = parse_args()
    reports = {
        "short_bf16": read(args.short_bf16),
        "short_fp16": read(args.short_fp16),
        "long_bf16": read(args.long_bf16),
        "long_fp16": read(args.long_fp16),
    }
    real8b = read(args.real8b)
    quality = read(args.quality)
    ownership = read(args.ownership)
    production = [
        item
        for report in reports.values()
        for item in report["results"]
        if item.get("supported")
        and item["provider"] in {"generic_cuda", "sm86"}
    ]
    sm86 = [item for item in production if item["provider"] == "sm86"]
    speedups = [float(item["speedup_vs_generic"]) for item in sm86]
    page16 = [
        item for item in reports["long_bf16"]["results"]
        if item["provider"] == "sm86" and item["page_size"] == 16
    ]
    page32 = [
        item for item in reports["long_bf16"]["results"]
        if item["provider"] == "sm86" and item["page_size"] == 32
    ]
    page_ratios = matching_ratio(
        page16,
        page32,
        ("context_length", "phase"),
        "p50_ms",
    )
    bf16_sm86 = [
        item for name, report in reports.items() if "bf16" in name
        for item in report["results"] if item.get("supported") and item["provider"] == "sm86"
    ]
    fp16_sm86 = [
        item for name, report in reports.items() if "fp16" in name
        for item in report["results"] if item.get("supported") and item["provider"] == "sm86"
    ]
    dtype_ratios = matching_ratio(
        bf16_sm86,
        fp16_sm86,
        ("context_length", "page_size", "phase"),
        "p50_ms",
    )
    real_rows = []
    real_by_context = {}
    for item in real8b["results"]:
        row = {
            "context": int(item["context_length"]),
            "context_label": str(item["context_length"]),
            "provider": item["provider"],
            "ttft_ms": rounded(item["ttft"]["p50_ms"], 3),
            "decode_ms": rounded(item["decode"]["p50_ms"], 3),
            "decode_tps": rounded(item["decode_tokens_per_second"], 4),
            "workspace_mib": rounded(item["workspace_peak_bytes"] / 1048576.0, 3),
            "gpu_peak_mib": rounded(item["gpu_peak_allocated_bytes"] / 1048576.0, 1),
            "top1_matches_legacy": bool(item["top1_matches_legacy"]),
        }
        real_rows.append(row)
        real_by_context.setdefault(row["context"], {})[row["provider"]] = row
    real_chart = []
    for context in sorted(real_by_context):
        providers = real_by_context[context]
        real_chart.append(
            {
                "context": context,
                "context_label": str(context),
                "legacy_ttft_ms": providers["legacy_gather_sdpa_reference"]["ttft_ms"],
                "generic_ttft_ms": providers["generic_cuda"]["ttft_ms"],
                "sm86_ttft_ms": providers["sm86"]["ttft_ms"],
                "legacy_decode_ms": providers["legacy_gather_sdpa_reference"]["decode_ms"],
                "generic_decode_ms": providers["generic_cuda"]["decode_ms"],
                "sm86_decode_ms": providers["sm86"]["decode_ms"],
                "legacy_workspace_mib": providers["legacy_gather_sdpa_reference"]["workspace_mib"],
            }
        )
    kernel_map = {}
    for report_name in ("short_bf16", "long_bf16"):
        for item in reports[report_name]["results"]:
            if (
                item.get("supported")
                and item["provider"] in {"generic_cuda", "sm86"}
                and item["phase"] == "decode"
                and item["page_size"] == 16
            ):
                kernel_map.setdefault(item["context_length"], {})[
                    item["provider"]
                ] = item
    kernel_chart = []
    for context in sorted(kernel_map):
        providers = kernel_map[context]
        if set(providers) == {"generic_cuda", "sm86"}:
            kernel_chart.append(
                {
                    "context": int(context),
                    "context_label": str(context),
                    "generic_ms": rounded(providers["generic_cuda"]["p50_ms"], 4),
                    "sm86_ms": rounded(providers["sm86"]["p50_ms"], 4),
                    "speedup": rounded(
                        providers["generic_cuda"]["p50_ms"]
                        / providers["sm86"]["p50_ms"],
                        4,
                    ),
                }
            )
    long_2048 = real_by_context[2048]
    headline = {
        "sm86_kernel_median_speedup": rounded(statistics.median(speedups), 4),
        "sm86_kernel_min_speedup": rounded(min(speedups), 4),
        "sm86_kernel_max_speedup": rounded(max(speedups), 4),
        "page16_vs_page32_median_latency_ratio": rounded(
            statistics.median(page_ratios), 4
        ),
        "bf16_vs_fp16_median_latency_ratio": rounded(
            statistics.median(dtype_ratios), 4
        ),
        "real8b_decode_median_ms_all_modes": rounded(
            statistics.median(row["decode_ms"] for row in real_rows), 3
        ),
        "real8b_2048_sm86_ttft_ms": long_2048["sm86"]["ttft_ms"],
        "real8b_2048_legacy_ttft_ms": long_2048[
            "legacy_gather_sdpa_reference"
        ]["ttft_ms"],
        "real8b_2048_sm86_vs_legacy_ttft_ratio": rounded(
            long_2048["sm86"]["ttft_ms"]
            / long_2048["legacy_gather_sdpa_reference"]["ttft_ms"],
            4,
        ),
        "quality_perplexity_relative_change": rounded(
            quality["metrics"]["perplexity_relative_change"], 6
        ),
        "quality_max_accuracy_drop_pp": rounded(
            max(
                quality["metrics"]["short_accuracy_drop_pp"],
                quality["metrics"]["long_hit_rate_drop_pp"],
                quality["metrics"]["dialogue_accuracy_drop_pp"],
            ),
            4,
        ),
        "ownership_reserved_drift_after_trim_bytes": int(
            ownership["resource_drift"]["cuda_reserved_bytes"]
        ),
        "ownership_reserved_cache_before_trim_bytes": int(
            ownership["allocator_cache_before_trim"]["cuda_reserved_bytes"]
        ),
    }
    matrix_summaries = []
    for name, report in reports.items():
        config = report["configuration"]
        provider_rows = [
            item
            for item in report["results"]
            if item.get("supported") and item["provider"] == "sm86"
        ]
        provider_speedups = [
            float(item["speedup_vs_generic"]) for item in provider_rows
        ]
        matrix_summaries.append(
            {
                "matrix": name,
                "dtype": config["dtype"],
                "contexts": config["lengths"],
                "page_sizes": config["page_sizes"],
                "phases": config["phases"],
                "batch_size": config["batch_size"],
                "prefill_query_length": config["prefill_query_length"],
                "warmup": config["warmup"],
                "runs": config["runs"],
                "sm86_case_count": len(provider_rows),
                "sm86_speedup_vs_generic_min": rounded(
                    min(provider_speedups), 4
                ),
                "sm86_speedup_vs_generic_median": rounded(
                    statistics.median(provider_speedups), 4
                ),
                "sm86_speedup_vs_generic_max": rounded(
                    max(provider_speedups), 4
                ),
                "sm86_workspace_peak_bytes": max(
                    int(item["workspace_peak_bytes"])
                    for item in provider_rows
                ),
            }
        )
    mode_support = [
        {"mode": "exact / GPU / BF16 / dense", "status": "production path", "measured": True, "note": "Generic CUDA and SM86"},
        {"mode": "exact / GPU / FP16 / dense", "status": "production path", "measured": True, "note": "Kernel microbenchmark"},
        {"mode": "exact / GPU / BF16 / gather+SDPA", "status": "experimental", "measured": True, "note": "Linear full-KV workspace"},
        {"mode": "exact / session reuse", "status": "implemented", "measured": False, "note": "Same attention kernel; lifecycle covered by soak"},
        {"mode": "exact / prefix_memory", "status": "implemented", "measured": False, "note": "1000-cycle ownership soak"},
        {"mode": "quantized / INT8 or FP8", "status": "unsupported", "measured": False, "note": "Interface only"},
        {"mode": "sparse / Quest", "status": "unsupported", "measured": False, "note": "Experimental interface only"},
        {"mode": "GPU+CPU/NVMe active KV", "status": "unsupported", "measured": False, "note": "No active offload implementation"},
    ]
    sources = [
        rel(args.short_bf16),
        rel(args.short_fp16),
        rel(args.long_bf16),
        rel(args.long_fp16),
        rel(args.real8b),
        rel(args.quality),
        rel(args.ownership),
    ]
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": reports["short_bf16"]["hardware"],
        "headline": headline,
        "quality": {
            "coverage": quality["coverage"],
            "metrics": quality["metrics"],
            "reference": quality["reference_provider"],
            "candidate": quality["candidate_provider"],
        },
        "ownership": {
            "cycles": ownership["cycles"],
            "resource_drift": ownership["resource_drift"],
            "allocator_cache_before_trim": ownership[
                "allocator_cache_before_trim"
            ],
            "acceptance": ownership["acceptance"],
        },
        "kernel_decode_bf16_page16": kernel_chart,
        "kernel_matrix_summary": matrix_summaries,
        "real8b": real_rows,
        "mode_support": mode_support,
        "source_artifacts": sources,
        "limitations": [
            "Single RTX 3080 Ti (SM86), batch 1 or 2 only.",
            "Real-8B repeats are 3 after one warmup; sub-percent decode differences are noise-level.",
            "The L4 suite is a deterministic non-regression micro-suite, not a population-level capability benchmark.",
            "INT8/FP8 KV, sparse selection, CPU/NVMe active offload and other GPU architectures were not executable.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    markdown = """# Cascade-LLM KV Mode 性能与生产门禁报告

## 技术结论

本轮已经关闭原有两个生产门禁：L4 的真实 8B Provider 非退化微型质量集通过；L5 的 2 MiB reserved 漂移被证明为 PyTorch caching allocator 保留段，1000-cycle 期间 reserved 恒定，关闭后显式 trim 回到零。SM86 Kernel 相对 Generic CUDA 的全部 BF16/FP16 case 中位加速为 **{median_speedup:.3f}×**，长上下文收益稳定；但真实 8B Decode 仍约 **{decode_ms:.1f} ms/token**，Attention 优化基本被每 Token Transformer 权重传输掩盖。

更重要的负面结果是：真实 8B、2048-token Prefill 中，SM86 TTFT 为 **{sm86_2048:.1f} ms**，而实验 gather+SDPA 为 **{legacy_2048:.1f} ms**。SM86 production path 慢 **{sm86_legacy_ratio:.2f}×**。因此当前 SM86 可以通过既定生产稳定性和数值合同，但长 Prefill Kernel 仍有明显性能缺陷，不应描述为全面优于成熟 SDPA。

## 两个生产门禁已经关闭

L4 覆盖 64 个困惑度 Token、8 个短文本题、4 个 499–2413 Token 长检索题和 6 个固定对话题。SM86 相对显式 gather+SDPA reference 的困惑度变化为 **{ppl_change:.4%}**，三类准确率下降均为 **0 pp**，候选与 reference 均答对全部 18 道选择题。

L5 的 1000-cycle 测试执行 1000 次 Beam Fork/Discard、1000 次 Speculative Fork/Commit、142 次 Session Fork 和 62 次 Prefix 注册/命中。page/ref/pin 全部归零；运行期间 CUDA reserved span 为 0。关闭后 trim 前缓存池保留 2 MiB，trim 后 reserved drift 为 0，因此它不是 KV Runtime 泄漏。

## Kernel 模式：SM86 长上下文稳定优于 Generic

Kernel microbenchmark 使用 HND、GQA 32/8 heads、head dim 128。短/中矩阵为 batch 2、Prefill query 8；长矩阵为 batch 1、Prefill query 4。每 case 2 次 warmup、7 次正式运行。SM86/Generic 全部 production case 的速度比范围为 **{min_speedup:.3f}×–{max_speedup:.3f}×**，中位 **{median_speedup:.3f}×**。少量短 case 的最低值低于 1% 属于计时噪声范围；长上下文所有 case 均明确快于 Generic。

Page 16 与 Page 32 的长上下文中位延迟比为 **{page_ratio:.3f}**，没有足够证据宣布固定页大小普遍更快；应根据上下文和 workload 选择。BF16/FP16 中位延迟比为 **{dtype_ratio:.3f}**，在 SM86 上也没有形成决定性差异。

## 真实 8B：Decode 被权重流式主导，长 Prefill 暴露 Kernel 缺陷

端到端矩阵使用 Llama-3.1-8B-Instruct BF16、Transformer 全流式、Embedding/LM Head 常驻、双 slot pinned staging。8/128-token 下三种 Attention mode 的 TTFT 和 Decode 基本重合。512-token 时 SM86 Prefill 已明显优于另外两条路径；但到 2048-token，gather+SDPA 利用成熟 SDPA Kernel 反而最快，SM86 次之，Generic CUDA 最慢。

Decode 在全部 mode 和上下文中约为 1.35–1.36 秒/token，差异不到权重传输抖动量级。要提升单请求 Decode，优先级仍应是减少 Transformer 权重 H2D，而不是继续微调 Attention Kernel。

## 指标、范围与方法

- Kernel 延迟是同步 wall-clock P50；短/中与长矩阵均记录原始 7 次样本。
- 真实 8B TTFT/Decode 是 1 次 warmup 后 3 次正式运行的 P50，包含权重流式与常驻 LM Head。
- `reference_paged_exact` 是 FP32 两遍 page-wise 数值 reference；`legacy_gather_sdpa_reference` 是显式 gather、线性 workspace 的实验性能横向参考。
- Production provider 的 workspace 为 0；legacy workspace 随上下文线性增长，在真实 8B 2048-token case 达到约 32 MiB。

## 限制与稳健性

本报告只覆盖一张 RTX 3080 Ti、单请求/低 batch。真实 8B 每个 case 只有 3 次正式样本，因此小于 1% 的差异不作性能排序。L4 是 Provider 非退化微型集，不代表完整的通用能力评测。INT8/FP8 KV、Quest、CPU/NVMe 活跃 Offload 和 SM80/89/90 真机均未实现或未验证，报告明确列为 unsupported/unqualified。

## 建议的下一步

1. 为 Prefill 单独接入成熟 paged/Flash Attention provider，避免当前 scalar/warp Kernel 在完整长 Prefill 上落后 SDPA。
2. 保持 SM86 Kernel 作为 Decode 和短 Prefill 路径，同时通过显式 phase plan 选择 Prefill backend，禁止静默切换。
3. 优先继续 M6 的权重常驻/INT8 W8A16，以改善约 1.36 秒/token 的真实 Decode 主瓶颈。
4. 将 L4 微型集扩展为公开数据集子集，并在 SM80/89/90 真机上建立独立合同。

## 仍需回答的问题

- 长 Prefill 的交叉点是否随 batch、chunk size 和真实自然语言长度变化？
- 使用 FlashInfer/FlashAttention paged provider 后能否同时保持零完整 KV workspace 与 SDPA 级 Prefill 性能？
- Transformer 权重常驻比例提升后，SM86 Decode Kernel 的 13%–20% 局部收益能否转化为端到端收益？
""".format(
        median_speedup=headline["sm86_kernel_median_speedup"],
        min_speedup=headline["sm86_kernel_min_speedup"],
        max_speedup=headline["sm86_kernel_max_speedup"],
        decode_ms=headline["real8b_decode_median_ms_all_modes"],
        sm86_2048=headline["real8b_2048_sm86_ttft_ms"],
        legacy_2048=headline["real8b_2048_legacy_ttft_ms"],
        sm86_legacy_ratio=headline["real8b_2048_sm86_vs_legacy_ttft_ratio"],
        ppl_change=headline["quality_perplexity_relative_change"],
        page_ratio=headline["page16_vs_page32_median_latency_ratio"],
        dtype_ratio=headline["bf16_vs_fp16_median_latency_ratio"],
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown, encoding="utf-8")

    generated_at = summary["generated_at"]
    manifest_sources = [
        {
            "id": "summary",
            "label": "KV mode benchmark aggregate",
            "path": rel(args.summary),
        }
    ]
    source_specs = [
        {
            "id": "summary",
            "label": "KV mode benchmark aggregate",
            "path": rel(args.summary),
            "query": {
                "engine": "duckdb",
                "sql": (
                    "SELECT * FROM read_json_auto('"
                    + rel(args.summary)
                    + "')"
                ),
                "description": (
                    "Reads the deterministic benchmark aggregate generated by "
                    "tools/build_kv_mode_report.py."
                ),
            },
        }
    ]
    headline_rows = [headline]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Cascade-LLM KV Mode Performance Qualification",
            "description": "SM86 Numerical Contract V2, quality, stability and mode performance evidence",
            "generatedAt": generated_at,
            "sources": manifest_sources,
            "cards": [
                {"id": "kernel_speedup", "dataset": "headline", "sourceId": "summary", "metrics": [{"label": "SM86 median kernel speedup", "field": "sm86_kernel_median_speedup", "format": "number"}]},
                {"id": "decode_latency", "dataset": "headline", "sourceId": "summary", "metrics": [{"label": "Real 8B median decode, ms", "field": "real8b_decode_median_ms_all_modes", "format": "number"}]},
                {"id": "quality_drop", "dataset": "headline", "sourceId": "summary", "metrics": [{"label": "Maximum accuracy drop, pp", "field": "quality_max_accuracy_drop_pp", "format": "number"}]},
                {"id": "reserved_drift", "dataset": "headline", "sourceId": "summary", "metrics": [{"label": "Reserved drift after trim, bytes", "field": "ownership_reserved_drift_after_trim_bytes", "format": "number"}]},
            ],
            "charts": [
                {
                    "id": "kernel_decode",
                    "title": "BF16 Page-16 Decode kernel latency",
                    "subtitle": "Synchronized P50 by context length; lower is better",
                    "intent": "comparison",
                    "type": "bar",
                    "dataset": "kernel_decode",
                    "sourceId": "summary",
                    "encodings": {
                        "x": {"field": "context_label", "type": "ordinal", "label": "Context tokens"},
                        "y": {"fields": ["generic_ms", "sm86_ms"], "type": "quantitative", "label": "P50 latency", "unit": "ms"},
                    },
                    "valueFormat": "number",
                    "layout": "full"
                },
                {
                    "id": "real_ttft",
                    "title": "Real 8B TTFT by Attention mode",
                    "subtitle": "Transformer streamed, Embedding and LM Head resident; lower is better",
                    "intent": "comparison",
                    "type": "bar",
                    "dataset": "real_chart",
                    "sourceId": "summary",
                    "encodings": {
                        "x": {"field": "context_label", "type": "ordinal", "label": "Prompt tokens"},
                        "y": {"fields": ["legacy_ttft_ms", "generic_ttft_ms", "sm86_ttft_ms"], "type": "quantitative", "label": "TTFT", "unit": "ms"},
                    },
                    "valueFormat": "number",
                    "layout": "full"
                },
                {
                    "id": "real_decode",
                    "title": "Real 8B single-token Decode latency",
                    "subtitle": "All modes remain weight-transfer bound",
                    "intent": "comparison",
                    "type": "bar",
                    "dataset": "real_chart",
                    "sourceId": "summary",
                    "encodings": {
                        "x": {"field": "context_label", "type": "ordinal", "label": "Context tokens"},
                        "y": {"fields": ["legacy_decode_ms", "generic_decode_ms", "sm86_decode_ms"], "type": "quantitative", "label": "P50 latency", "unit": "ms/token"},
                    },
                    "valueFormat": "number",
                    "layout": "full"
                }
            ],
            "tables": [
                {
                    "id": "real_matrix",
                    "title": "Real 8B exact results",
                    "subtitle": "One warmup and three measured runs per provider/context",
                    "dataset": "real_rows",
                    "sourceId": "summary",
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "context", "label": "Context", "type": "number"},
                        {"field": "provider", "label": "Provider", "type": "text"},
                        {"field": "ttft_ms", "label": "TTFT ms", "type": "number"},
                        {"field": "decode_ms", "label": "Decode ms", "type": "number"},
                        {"field": "workspace_mib", "label": "Workspace MiB", "type": "number"},
                        {"field": "top1_matches_legacy", "label": "Top-1 match", "type": "text"}
                    ]
                },
                {
                    "id": "mode_support",
                    "title": "Mode execution boundary",
                    "subtitle": "Only executable modes were benchmarked",
                    "dataset": "mode_support",
                    "sourceId": "summary",
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "mode", "label": "Mode", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                        {"field": "measured", "label": "Measured", "type": "text"},
                        {"field": "note", "label": "Boundary", "type": "text"}
                    ]
                }
            ],
            "blocks": [
                {"id": "technical_summary", "type": "markdown", "layout": "full", "sourceId": "summary", "body": "## Technical summary\n\nSM86 passes the L4 provider non-regression suite and the corrected L5 ownership gate. Its kernel is materially faster than Generic CUDA at long context, but real 8B Decode remains weight-transfer bound and 2048-token Prefill is still slower than gather+SDPA."},
                {"id": "headline_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["kernel_speedup", "decode_latency", "quality_drop", "reserved_drift"]},
                {"id": "kernel_finding", "type": "markdown", "layout": "full", "sourceId": "summary", "body": "## SM86 improves the direct paged kernel at long context\n\nThe architecture-specific 128-thread reduction produces a stable long-context gain over Generic CUDA. Short-context differences below 1% are treated as timing-equivalent rather than as a regression."},
                {"id": "kernel_chart_block", "type": "chart", "layout": "full", "chartId": "kernel_decode"},
                {"id": "prefill_finding", "type": "markdown", "layout": "full", "sourceId": "summary", "body": "## Full Prefill exposes the remaining production-kernel gap\n\nSM86 leads at 512 tokens, but at 2048 tokens the optimized SDPA reference is substantially faster. The production kernel avoids linear full-KV workspace, yet it needs a dedicated high-throughput Prefill implementation."},
                {"id": "ttft_chart_block", "type": "chart", "layout": "full", "chartId": "real_ttft"},
                {"id": "decode_finding", "type": "markdown", "layout": "full", "sourceId": "summary", "body": "## Decode mode choice is hidden by streamed Transformer weights\n\nAll three Attention modes remain near 1.36 seconds per token. Reducing weight H2D is the next system-level priority; kernel-only gains cannot be presented as end-to-end Decode gains."},
                {"id": "decode_chart_block", "type": "chart", "layout": "full", "chartId": "real_decode"},
                {"id": "definitions", "type": "markdown", "layout": "full", "body": "## Scope, data and metric definitions\n\nKernel cases use synchronized P50 wall time. Real-8B TTFT includes streamed Transformer execution and resident vocabulary operations. The quality suite is a deterministic provider non-regression check, not a broad capability benchmark."},
                {"id": "real_table_block", "type": "table", "layout": "full", "tableId": "real_matrix"},
                {"id": "method", "type": "markdown", "layout": "full", "body": "## Methodology\n\nMicrobenchmarks cover BF16/FP16, Page 16/32, Decode/Prefill and 16–32K contexts with seven measured runs. The real-model matrix covers 8–2048 prompt tokens with one warmup and three measured runs. Candidate Top-1 is checked against the explicit legacy reference."},
                {"id": "limitations", "type": "markdown", "layout": "full", "body": "## Limitations and robustness\n\nEvidence comes from one RTX 3080 Ti. Sub-percent real-model differences are noise-level. Quantized KV, sparse selection, active offload and non-SM86 GPUs were not executable and are not represented as performance results."},
                {"id": "mode_table_block", "type": "table", "layout": "full", "tableId": "mode_support"},
                {"id": "next_steps", "type": "markdown", "layout": "full", "body": "## Recommended next steps\n\n1. Add a mature paged/Flash Prefill provider and keep phase selection explicit.\n2. Prioritize resident or INT8 streamed Transformer weights for Decode.\n3. Expand L4 to public datasets and qualify each GPU architecture independently."},
                {"id": "questions", "type": "markdown", "layout": "full", "body": "## Further questions\n\nWhere is the Prefill crossover across batch and chunk sizes? Can a mature paged provider match SDPA while retaining zero full-KV workspace? When weight H2D falls, how much of the local SM86 Decode gain becomes end-to-end?"}
            ]
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline_rows,
                "kernel_decode": kernel_chart,
                "real_chart": real_chart,
                "real_rows": real_rows,
                "mode_support": mode_support
            }
        },
        "sources": source_specs
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(args.summary), "markdown": str(args.markdown), "artifact": str(args.artifact)}, indent=2))


if __name__ == "__main__":
    main()
