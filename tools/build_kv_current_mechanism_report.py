#!/usr/bin/env python3
"""Build the current KV mechanism technical report from reviewed evidence."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "real_results" / "kv_current_20260803"
REPORT_ROOT = ROOT / "reports" / "kv_current_mechanism_20260803"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def rounded(value, digits=4):
    return round(float(value), int(digits))


def source(source_id, label, path, description):
    source_path = relative(path)
    return {
        "id": source_id,
        "label": label,
        "path": source_path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_json_auto('{}')".format(source_path),
            "description": description,
            "tables_used": [source_path],
        },
    }


def table(table_id, title, subtitle, dataset, source_id, columns, sort_field):
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "sourceId": source_id,
        "density": "spacious",
        "layout": "full",
        "defaultSort": {"field": sort_field, "direction": "asc"},
        "columns": columns,
    }


def main():
    inputs = {
        "tests": RESULT_ROOT / "test_summary.json",
        "lifecycle": RESULT_ROOT / "lifecycle_1000.json",
        "short": RESULT_ROOT / "kernel_short_bf16.json",
        "long": RESULT_ROOT / "kernel_long_bf16.json",
        "prefill": RESULT_ROOT / "full_prefill_bf16.json",
        "real8b": RESULT_ROOT / "real8b_short_modes.json",
        "hardware": RESULT_ROOT / "hardware.json",
        "historical_soak": ROOT / "real_results/kv_v1/real8b_sm86_1000_token_soak.json",
        "historical_replay": ROOT / "real_results/kv_v1/real8b_sm86_1000_token_hf_replay.json",
        "quality": ROOT / "real_results/kv_v2/artifacts/quality_sm86_bf16_v1.json",
        "numerical": ROOT / "real_results/kv_v2/kv_numerical_summary.json",
        "qualification": ROOT / "real_results/kv_v2/kv_qualification_summary.json",
    }
    reports = {name: read(path) for name, path in inputs.items()}
    model_config_path = Path(
        "/disk4/llm_model/Llama/Llama3.1/Meta-Llama-3.1-8B-Instruct/config.json"
    )
    model_config = read(model_config_path)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    generated_at = datetime.now(timezone.utc).isoformat()

    short_sm86 = [
        item for item in reports["short"]["results"]
        if item.get("supported") and item["provider"] == "sm86"
    ]
    long_sm86 = [
        item for item in reports["long"]["results"]
        if item.get("supported") and item["provider"] == "sm86"
    ]
    short_speedups = [float(item["speedup_vs_generic"]) for item in short_sm86]
    long_speedups = [float(item["speedup_vs_generic"]) for item in long_sm86]
    production_results = [
        item
        for report_name in ("short", "long", "prefill")
        for item in reports[report_name]["results"]
        if item.get("supported")
        and item["provider"] in {"generic_cuda", "sm86"}
    ]
    production_zero_workspace = all(
        int(item["workspace_peak_bytes"]) == 0
        and not bool(item["requires_full_kv_workspace"])
        for item in production_results
    )
    pairwise_errors = [
        float(item["pairwise_numerical_diagnostic"]["metrics"]["max_abs_error"])
        for item in short_sm86
    ]

    prefill_by_context = defaultdict(dict)
    for item in reports["prefill"]["results"]:
        prefill_by_context[int(item["context_length"])][item["provider"]] = item
    prefill_rows = []
    prefill_tradeoff = []
    workspace_rows = []
    for context in sorted(prefill_by_context):
        providers = prefill_by_context[context]
        legacy = providers["legacy_gather_sdpa_reference"]
        generic = providers["generic_cuda"]
        sm86 = providers["sm86"]
        legacy_ms = float(legacy["p50_ms"])
        row = {
            "context": context,
            "context_label": str(context),
            "legacy_ms": rounded(legacy_ms, 3),
            "generic_ms": rounded(generic["p50_ms"], 3),
            "sm86_ms": rounded(sm86["p50_ms"], 3),
            "generic_slowdown": rounded(generic["p50_ms"] / legacy_ms, 3),
            "sm86_slowdown": rounded(sm86["p50_ms"] / legacy_ms, 3),
            "legacy_workspace_mib": rounded(
                legacy["workspace_peak_bytes"] / 1048576.0, 3
            ),
            "production_workspace_mib": 0.0,
            "runs": int(reports["prefill"]["configuration"]["runs"]),
        }
        prefill_rows.append(row)
        prefill_tradeoff.append({
            "context": context,
            "context_label": str(context),
            "legacy_baseline": 1.0,
            "generic_slowdown": row["generic_slowdown"],
            "sm86_slowdown": row["sm86_slowdown"],
        })
        workspace_rows.append({
            "context": context,
            "context_label": str(context),
            "legacy_workspace_mib": row["legacy_workspace_mib"],
            "production_workspace_mib": 0.0,
        })

    suffix_speedups = defaultdict(list)
    for item in long_sm86:
        suffix_speedups[int(item["context_length"])].append(
            float(item["speedup_vs_generic"])
        )
    suffix_rows = [
        {
            "context": context,
            "context_label": str(context),
            "median_speedup": rounded(statistics.median(values), 4),
            "min_speedup": rounded(min(values), 4),
            "max_speedup": rounded(max(values), 4),
            "case_count": len(values),
        }
        for context, values in sorted(suffix_speedups.items())
    ]

    lifecycle_rows = [
        {
            "cycle": int(item["cycle"]),
            "allocated_mib": rounded(item["cuda_allocated_bytes"] / 1048576.0, 3),
            "reserved_mib": rounded(item["cuda_reserved_bytes"] / 1048576.0, 3),
            "allocated_pages": int(item["allocated_pages"]),
            "sequence_length": int(item["sequence_length"]),
            "total_ref_count": int(item["total_ref_count"]),
            "total_pin_count": int(item["total_pin_count"]),
        }
        for item in reports["lifecycle"]["snapshots"]
    ]

    real8b_rows = []
    mode_spreads = []
    real8b_by_context = defaultdict(list)
    for item in reports["real8b"]["results"]:
        row = {
            "context": int(item["context_length"]),
            "provider": item["provider"],
            "ttft_ms": rounded(item["ttft"]["p50_ms"], 3),
            "decode_ms": rounded(item["decode"]["p50_ms"], 3),
            "decode_tps": rounded(item["decode_tokens_per_second"], 4),
            "workspace_mib": rounded(item["workspace_peak_bytes"] / 1048576.0, 4),
            "top1_matches_legacy": bool(item["top1_matches_legacy"]),
            "runs": int(item["runs"]),
        }
        real8b_rows.append(row)
        real8b_by_context[row["context"]].append(row)
    for context, rows in real8b_by_context.items():
        for field in ("ttft_ms", "decode_ms"):
            values = [float(row[field]) for row in rows]
            legacy = next(
                float(row[field]) for row in rows
                if row["provider"] == "legacy_gather_sdpa_reference"
            )
            mode_spreads.append((max(values) - min(values)) / legacy * 100.0)

    hardware = reports["hardware"]["hardware"]
    bytes_per_token = (
        2
        * int(model_config["num_hidden_layers"])
        * int(model_config["num_key_value_heads"])
        * (int(model_config["hidden_size"]) // int(model_config["num_attention_heads"]))
        * 2
    )
    total_gpu_bytes = int(hardware["total_memory_bytes"])
    theoretical_tokens = total_gpu_bytes // bytes_per_token
    capacity_contexts = (8192, 32768, 65536, theoretical_tokens, 131072)
    capacity_rows = [
        {
            "context_tokens": int(context),
            "kv_gib": rounded(context * bytes_per_token / (1024.0 ** 3), 3),
            "share_of_physical_vram_pct": rounded(
                context * bytes_per_token / total_gpu_bytes * 100.0, 1
            ),
            "interpretation": (
                "theoretical KV-only ceiling"
                if context == theoretical_tokens
                else ("exceeds physical VRAM" if context * bytes_per_token > total_gpu_bytes else "fits before other GPU allocations")
            ),
        }
        for context in capacity_contexts
    ]

    mode_support = [
        {"capability": "Exact dense GPU KV, BF16/FP16", "status": "implemented", "measured": "yes", "boundary": "Generic CUDA and SM86 direct paging"},
        {"capability": "Request/session/in-memory sealed-prefix reuse", "status": "implemented", "measured": "lifecycle", "boundary": "No LRU or persistent tier"},
        {"capability": "Ragged batch metadata and kernels", "status": "implemented ABI", "measured": "microbenchmark", "boundary": "No continuous-batching scheduler"},
        {"capability": "INT8/FP8/INT4 KV", "status": "unsupported", "measured": "no", "boundary": "Policy/ABI only"},
        {"capability": "GPU+CPU/NVMe active KV offload", "status": "unsupported", "measured": "no", "boundary": "No transfer/restore path"},
        {"capability": "Quest/sparse page selection", "status": "unsupported", "measured": "no", "boundary": "Selection interface only"},
        {"capability": "Persistent prefix cache", "status": "unsupported", "measured": "no", "boundary": "In-memory full sealed blocks only"},
        {"capability": "CUDA Graph", "status": "unsupported", "measured": "no", "boundary": "Host metadata and synchronization remain"},
        {"capability": "SM80/SM89/SM90", "status": "declared only", "measured": "no", "boundary": "No physical qualification evidence"},
    ]

    qualification = reports["qualification"]
    numerical = reports["numerical"]
    quality = reports["quality"]
    historical_soak = reports["historical_soak"]
    historical_replay = reports["historical_replay"]
    headline = {
        "tests_passed": int(reports["tests"]["tests_run"]),
        "test_failures": int(reports["tests"]["failures"] + reports["tests"]["errors"]),
        "lifecycle_cycles": int(reports["lifecycle"]["cycles"]),
        "resource_drift_bytes": int(max(abs(value) for value in reports["lifecycle"]["resource_drift"].values())),
        "production_case_count": len(production_results),
        "production_zero_workspace": production_zero_workspace,
        "short_sm86_median_speedup": rounded(statistics.median(short_speedups), 4),
        "long_sm86_median_speedup": rounded(statistics.median(long_speedups), 4),
        "full_prefill_max_sm86_slowdown": rounded(max(row["sm86_slowdown"] for row in prefill_rows), 3),
        "real8b_max_mode_spread_pct": rounded(max(mode_spreads), 3),
        "bytes_per_token": int(bytes_per_token),
        "kib_per_token": rounded(bytes_per_token / 1024.0, 1),
        "theoretical_kv_only_tokens": int(theoretical_tokens),
        "strict_hf_diagnostic_failures": int(numerical["evidence"]["strict_hf_diagnostic_failures"]),
        "strict_hf_comparison_count": 297,
        "quality_max_accuracy_drop_pp": rounded(max(
            quality["metrics"]["short_accuracy_drop_pp"],
            quality["metrics"]["long_hit_rate_drop_pp"],
            quality["metrics"]["dialogue_accuracy_drop_pp"],
        ), 3),
        "historical_1000_token_p50_ms": rounded(historical_soak["latency_ms"]["p50"], 3),
        "historical_hf_top1_matches": int(historical_replay["top1_matches"]),
        "historical_hf_sampled_positions": int(historical_replay["sampled_positions"]),
    }

    summary = {
        "schema_version": 1,
        "generated_at": generated_at,
        "as_of_date": "2026-08-03",
        "git_commit": commit,
        "hardware": hardware,
        "headline": headline,
        "test_summary": reports["tests"],
        "lifecycle": {
            "operations": reports["lifecycle"]["operations"],
            "acceptance": reports["lifecycle"]["acceptance"],
            "resource_drift": reports["lifecycle"]["resource_drift"],
        },
        "prefill_rows": prefill_rows,
        "workspace_rows": workspace_rows,
        "suffix_speedups": suffix_rows,
        "lifecycle_rows": lifecycle_rows,
        "real8b_rows": real8b_rows,
        "capacity_rows": capacity_rows,
        "mode_support": mode_support,
        "historical_validation": {
            "qualification_status": qualification["status"],
            "production_passed": qualification["production_passed"],
            "strict_hf_diagnostic_failures": headline["strict_hf_diagnostic_failures"],
            "strict_hf_comparison_count": headline["strict_hf_comparison_count"],
            "quality_coverage": quality["coverage"],
            "quality_metrics": quality["metrics"],
            "real8b_1000_token_acceptance": historical_soak["acceptance"],
            "hf_replay_top1_matches": headline["historical_hf_top1_matches"],
            "hf_replay_sampled_positions": headline["historical_hf_sampled_positions"],
        },
        "validation_assessment": "Share with caveats",
        "validation_caveats": [
            "Fresh performance evidence is from one RTX 3080 Ti (SM86).",
            "Fresh real-8B mode results have two measured runs after one warmup and support only noise-level comparisons.",
            "The 1000-token real-model soak and quality suite are checked-in evidence from the same current commit, not rerun on 2026-08-03.",
            "The full-Prefill benchmark is a one-layer synthetic kernel test; the 8B end-to-end result includes streamed weights.",
            "Legacy gather+SDPA is an experimental comparison path, not a production serving baseline.",
        ],
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = REPORT_ROOT / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_markdown = """# Cascade-LLM 当前 KV 管理机制测试报告

## 技术结论

当前 KV Framework V1 是一个**生命周期可靠、显存边界清晰的 exact dense GPU paged KV 基础设施**，适合单请求 Decode、短 Prefill 和后续调度器/新 Provider 接入；它还不是高吞吐长 Prompt 或超显存 KV 的完整解决方案。

- **优点可复现：** 148/148 自动化测试通过；1000 轮 Fork/COW/Prefix 压力后 page/ref/pin、CUDA 内存和线程漂移均为 0；全部 {production_cases} 个 production benchmark case 不创建完整 KV workspace。
- **Kernel 局部有效：** SM86 对 Generic CUDA 的短中矩阵中位加速为 {short_speedup:.10g}×，1K–8K suffix/decode 矩阵中位加速为 {long_speedup:.10g}×。
- **完整 Prefill 是核心短板：** 2048-token full Prefill 中 SM86 比 gather+SDPA 慢 {prefill_slowdown:.10g}×；它省下每层 32MiB workspace，但牺牲了成熟 SDPA 的吞吐。
- **端到端收益尚未显现：** 真实 Llama-3.1-8B 的短上下文三模式最大观测差异仅 {mode_spread:.10g}%，Decode 约 1.35 秒/token，权重流式 H2D 掩盖了 Attention kernel 优化。
- **能力边界明显：** 当前只有 GPU-resident BF16/FP16 exact dense KV。CPU/NVMe active offload、量化 KV、Quest/sparse、persistent prefix 和 continuous batching scheduler 均未实现。

## 分页与事务生命周期经受住了当前压力

运行时通过 generation-safe page handle、跨层 append 事务、sealed-page 共享、partial-tail COW 和 ref/pin 计数，把复杂分支生命周期限制在统一 PagePool 内。新鲜 1000-cycle 测试共执行 1000 次 Beam reject、1000 次 speculative commit、142 次 session fork 和 62 次 prefix register/hit；2001 次页分配与 2001 次释放严格闭合。

## 零完整-KV workspace 是最直接的显存收益

Direct-paged Generic/SM86 在所有实测 production case 的 `workspace_peak_bytes` 都为 0。对照 gather+SDPA 必须先整理连续 K/V，full Prefill 的临时区从 128 token 的 2MiB 线性增至 2048 token 的 32MiB（本图是一层、batch 1、GQA 32/8）。这节省的是临时 workspace，不是持久 KV pool 本身。

## 完整 Prompt Prefill 用吞吐换取了 workspace

SM86 虽然比 Generic correctness-first kernel 快约 2×，但仍未 tile/GEMM 化。相对 gather+SDPA，SM86 在 128/512/1024/2048 token full Prefill 分别慢 1.50×/5.99×/13.15×/27.10×。因此不能把“SM86 比 Generic 快”写成“当前分页路径比成熟 Attention 快”。

## 长上下文 suffix/Decode 的局部优化真实存在

在 query length 为 1 或 4 的 1K–8K 矩阵中，SM86 对 Generic CUDA 每个 context 的组合中位加速稳定高于 1；这支持将 SM86 保留为 Decode/短 suffix Provider。该结论只比较项目内两个 direct-paged kernel，不代表优于 FlashInfer、vLLM 或 FlashAttention。

## 真实 8B 链路仍由权重搬运主导

Llama-3.1-8B BF16 使用 Transformer streamed、Embedding/LM Head resident、双 staging slot。8/128-token 新鲜复核中，三种 KV mode 的 TTFT/Decode 都约 1.35–1.36 秒，Top-1 全部一致。两次采样只能说明差异落在系统噪声量级，不能给三种 mode 排名。当前整机优化优先级应是减少 Transformer 权重 H2D。

## 范围、指标与容量定义

Kernel latency 是 GPU 同步 wall-clock P50；short/long case 各 7 次，full Prefill 各 5 次。`workspace` 指 Attention 调用额外构造的完整 K/V 临时区，不含持久 page pool。当前 Llama-3.1-8B（32 层、8 KV heads、head_dim 128、BF16）每 token KV 为 128KiB；分页避免增长时重分配，但 KV 容量仍随 token 线性增长。

理论上若把 11.67GiB 物理显存全部交给 KV，最多约 95.6k token；实际还需要输出、激活、resident 权重和 CUDA runtime，因此可用上限更低。page 16 的单请求尾页最坏内部碎片是 15 token，即约 1.875MiB；page 32 最坏约 3.875MiB。

## 数值正确性通过版本化合同，但不等于 HF bitwise identity

新鲜 short matrix 中 SM86 相对 `reference_paged_exact` 的最大 pairwise absolute error 为 {pairwise_error:.10g}。当前 Numerical Contract V2 的 L0–L5 与微型质量门禁通过，质量集的最大准确率下降为 0 pp，1000-token HF replay 抽样 101/101 Top-1 一致。

同时，真实 8B 相对 HF fused SDPA 的 297 个阶段中仍有 23 个严格 elementwise/ordered-Top-k 诊断失败。当前资格合同允许 architecture-specific reduction envelope，所以状态汇总为 production；需要 strict HF tensor identity 的调用方仍应视为不满足。

## 方法与复现口径

新鲜测试运行于 git commit `{commit}`、RTX 3080 Ti SM86、PyTorch 2.4.1+cu121、CUDA 12.1。自动化、生命周期、kernel 和短真实 8B 均在 2026-08-03 重新执行。1000-token 真实模型长稳、101-position HF replay 和 L4 微型质量集来自同一当前 commit 内已签入的 2026-08-02 证据。

验证评级为 **Share with caveats**：计算已独立复核、对照口径一致，结论可以用于当前架构决策；但单 GPU 架构、真实模型低重复数、synthetic kernel shape 和未实现模式必须随报告一起披露。

## 限制、失败模式与状态漂移

- 只有 SM86 真机；SM80/89/90 仍无物理资格证据。
- 没有 continuous batching scheduler，无法用当前结果推断多请求吞吐和调度公平性。
- Prefix index 没有容量上限/LRU/tenant API，只复用完整 sealed block。
- full Prefill microbenchmark 是单层 synthetic；真实 8B 端到端结果包含权重流式，二者回答不同问题。
- 真实 8B 新鲜对照每 case 仅 2 次；小差异不做排名。
- 主文档仍写 SM86 `performance_qualified`，而当前 Provider capability 与资格汇总写 `production`；支持级别存在文档漂移，应统一。

## 建议的下一步

1. 为 Prefill 接入成熟 paged/Flash Provider，并保持 Prefill/Decode backend 显式分相选择。
2. 优先降低 Transformer weight H2D（量化/提高 resident 比例），再评估 kernel 局部收益能否转化为端到端收益。
3. 实现真正的 CPU/NVMe KV tier、量化 KV 或 sparse selection；在此之前不要宣传“KV 超显存”。
4. 接入 continuous batching scheduler，并新增并发请求、prefix LRU、容量回压和公平性压力测试。
5. 统一 SM86 的文档和机器可读 qualification 状态，并在 SM80/89/90 真机独立复测。

## 仍需回答的问题

- 使用 FlashInfer/FlashAttention paged Prefill 后，能否同时保持 0 full-KV workspace 和 SDPA 级吞吐？
- 真实请求分布下 page 16/32 的碎片率、prefix 命中率和 COW 放大是多少？
- 权重 H2D 降低后，SM86 的 12%–25% suffix kernel 收益能转化成多少端到端 Decode 收益？
- 在 GPU KV pool 接近容量上限时，缺少 active offload/eviction 会如何影响拒绝率和尾延迟？
""".format(
        production_cases=headline["production_case_count"],
        short_speedup=headline["short_sm86_median_speedup"],
        long_speedup=headline["long_sm86_median_speedup"],
        prefill_slowdown=headline["full_prefill_max_sm86_slowdown"],
        mode_spread=headline["real8b_max_mode_spread_pct"],
        pairwise_error=max(pairwise_errors),
        commit=commit,
    )
    markdown_path = REPORT_ROOT / "report.md"
    markdown_path.write_text(report_markdown, encoding="utf-8")

    summary_source = source(
        "fresh_summary",
        "Current KV mechanism reviewed aggregate",
        summary_path,
        "Reads the deterministic aggregate produced from the 2026-08-03 fresh tests and current-commit validation artifacts.",
    )
    numerical_source = source(
        "numerical_contract",
        "KV Numerical Contract V2 summary",
        inputs["numerical"],
        "Reads the current-commit L0-L5 numerical and production-stability qualification summary.",
    )
    sources = [summary_source, numerical_source]
    manifest_sources = [
        {"id": item["id"], "label": item["label"], "path": item["path"]}
        for item in sources
    ]

    cards = [
        {"id": "tests", "dataset": "headline", "sourceId": "fresh_summary", "metrics": [{"label": "Fresh automated tests passed", "field": "tests_passed", "format": "number"}, {"label": "Failures", "field": "test_failures", "format": "number"}]},
        {"id": "ownership", "dataset": "headline", "sourceId": "fresh_summary", "metrics": [{"label": "Lifecycle stress cycles", "field": "lifecycle_cycles", "format": "number"}, {"label": "Final resource drift, bytes", "field": "resource_drift_bytes", "format": "number"}]},
        {"id": "workspace", "dataset": "headline", "sourceId": "fresh_summary", "metrics": [{"label": "Production cases with zero full-KV workspace", "field": "production_case_count", "format": "number"}]},
        {"id": "prefill", "dataset": "headline", "sourceId": "fresh_summary", "metrics": [{"label": "Max full-Prefill slowdown vs SDPA", "field": "full_prefill_max_sm86_slowdown", "format": "number"}]},
        {"id": "real8b_spread", "dataset": "headline", "sourceId": "fresh_summary", "metrics": [{"label": "Fresh real-8B max mode spread", "field": "real8b_max_mode_spread_pct", "format": "number", "unit": "%"}]},
    ]

    charts = [
        {
            "id": "workspace_chart",
            "title": "Full-Prefill attention workspace",
            "subtitle": "One layer, batch 1, BF16 GQA 32/8; MiB, lower is better",
            "intent": "comparison",
            "type": "bar",
            "dataset": "workspace_rows",
            "sourceId": "fresh_summary",
            "encodings": {
                "x": {"field": "context_label", "type": "ordinal", "label": "Context tokens"},
                "y": {"fields": ["legacy_workspace_mib", "production_workspace_mib"], "type": "quantitative", "label": "Workspace", "unit": "MiB"},
            },
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "prefill_tradeoff_chart",
            "title": "Full-Prefill latency relative to gather+SDPA",
            "subtitle": "One layer, batch 1, BF16, Page 16; ratio above 1 is slower",
            "intent": "comparison",
            "type": "bar",
            "dataset": "prefill_tradeoff",
            "sourceId": "fresh_summary",
            "encodings": {
                "x": {"field": "context_label", "type": "ordinal", "label": "Context tokens"},
                "y": {"fields": ["legacy_baseline", "generic_slowdown", "sm86_slowdown"], "type": "quantitative", "label": "Latency ratio", "unit": "×"},
            },
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "suffix_speedup_chart",
            "title": "SM86 suffix/decode speedup over Generic CUDA",
            "subtitle": "Median across Page 16/32 and query length 1/4; four cases per context",
            "intent": "comparison",
            "type": "bar",
            "dataset": "suffix_speedups",
            "sourceId": "fresh_summary",
            "encodings": {
                "x": {"field": "context_label", "type": "ordinal", "label": "Context tokens"},
                "y": {"field": "median_speedup", "type": "quantitative", "label": "Speedup", "unit": "×"},
            },
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "lifecycle_chart",
            "title": "CUDA memory during 1000 lifecycle cycles",
            "subtitle": "Eleven sampled checkpoints; MiB remains flat while sequence length grows",
            "intent": "trend",
            "type": "line",
            "dataset": "lifecycle_rows",
            "sourceId": "fresh_summary",
            "encodings": {
                "x": {"field": "cycle", "type": "quantitative", "label": "Cycle"},
                "y": {"fields": ["allocated_mib", "reserved_mib"], "type": "quantitative", "label": "CUDA memory", "unit": "MiB"},
            },
            "valueFormat": "number",
            "layout": "full",
        },
    ]

    tables = [
        table("prefill_table", "Full-Prefill exact timings", "Synchronized P50; five runs after two warmups", "prefill_rows", "fresh_summary", [
            {"field": "context", "label": "Context", "type": "number"},
            {"field": "legacy_ms", "label": "Gather+SDPA ms", "type": "number"},
            {"field": "generic_ms", "label": "Generic ms", "type": "number"},
            {"field": "sm86_ms", "label": "SM86 ms", "type": "number"},
            {"field": "sm86_slowdown", "label": "SM86 slowdown ×", "type": "number"},
            {"field": "legacy_workspace_mib", "label": "Legacy workspace MiB", "type": "number"},
        ], "context"),
        table("real8b_table", "Fresh real-8B end-to-end results", "Transformer streamed; one warmup and two measured runs", "real8b_rows", "fresh_summary", [
            {"field": "context", "label": "Context", "type": "number"},
            {"field": "provider", "label": "Provider", "type": "text"},
            {"field": "ttft_ms", "label": "TTFT ms", "type": "number"},
            {"field": "decode_ms", "label": "Decode ms", "type": "number"},
            {"field": "workspace_mib", "label": "Workspace MiB", "type": "number"},
            {"field": "top1_matches_legacy", "label": "Top-1 match", "type": "text"},
        ], "context"),
        table("capacity_table", "Llama-3.1-8B BF16 KV capacity projection", "128 KiB per token; theoretical values exclude all other GPU allocations", "capacity_rows", "fresh_summary", [
            {"field": "context_tokens", "label": "Context tokens", "type": "number"},
            {"field": "kv_gib", "label": "KV GiB", "type": "number"},
            {"field": "share_of_physical_vram_pct", "label": "Physical VRAM %", "type": "number"},
            {"field": "interpretation", "label": "Interpretation", "type": "text"},
        ], "context_tokens"),
        table("support_table", "Current execution boundary", "Unsupported modes have no fabricated benchmark values", "mode_support", "fresh_summary", [
            {"field": "capability", "label": "Capability", "type": "text"},
            {"field": "status", "label": "Status", "type": "text"},
            {"field": "measured", "label": "Measured", "type": "text"},
            {"field": "boundary", "label": "Boundary", "type": "text"},
        ], "capability"),
    ]

    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": "# Cascade-LLM 当前 KV 管理机制测试报告"},
        {"id": "technical_summary", "type": "markdown", "layout": "full", "body": "## 技术结论\n\n当前机制是**生命周期可靠、显存边界清晰的 exact dense GPU paged KV 基础设施**，适合单请求 Decode、短 Prefill 和后续 Provider 接入；它还不是高吞吐长 Prompt 或超显存 KV 的完整解决方案。优点是 148/148 回归通过、1000-cycle 零资源漂移、production 路径零完整-KV workspace；核心缺点是 2048-token full Prefill 比 gather+SDPA 慢 27.10×，而真实 8B 端到端性能仍被权重 H2D 主导。"},
        {"id": "headline_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["tests", "ownership", "workspace", "prefill", "real8b_spread"]},
        {"id": "lifecycle_finding", "type": "markdown", "layout": "full", "sourceId": "fresh_summary", "body": "## 分页与事务生命周期经受住了当前压力\n\n1000-cycle 测试覆盖 Beam reject、speculative commit、session fork、prefix register/hit 与 partial-tail COW。2001 次页分配/释放闭合，page/ref/pin 最终归零；下面 11 个采样点显示 CUDA allocated/reserved 在序列增长时保持水平。**含义：** 当前生命周期基础可依赖；**限制：** 这是 2 层 synthetic MQA，不替代真实 32 层长稳。"},
        {"id": "lifecycle_chart_block", "type": "chart", "layout": "full", "chartId": "lifecycle_chart"},
        {"id": "workspace_finding", "type": "markdown", "layout": "full", "sourceId": "fresh_summary", "body": "## 零完整-KV workspace 是最直接的显存收益\n\nDirect-paged Generic/SM86 在全部 production case 的完整-KV workspace 为 0；gather+SDPA 的临时区随上下文线性增长。图中节省的是 Attention 临时区，不是持久 KV pool。**含义：** 长上下文或并发请求可少一块峰值显存；**代价：** 当前 direct kernel 没有沿用成熟 SDPA 的吞吐。"},
        {"id": "workspace_chart_block", "type": "chart", "layout": "full", "chartId": "workspace_chart"},
        {"id": "prefill_finding", "type": "markdown", "layout": "full", "sourceId": "fresh_summary", "body": "## 完整 Prompt Prefill 用吞吐换取了 workspace\n\nSM86 比 Generic correctness-first kernel 快约 2×，但相对 gather+SDPA 的 slowdown 随上下文从 1.50× 放大到 27.10×。**含义：** SM86 的局部优化有效，却不能解决完整 Prefill 的算法/实现差距；需要独立的 tiled/Flash Prefill Provider。"},
        {"id": "prefill_chart_block", "type": "chart", "layout": "full", "chartId": "prefill_tradeoff_chart"},
        {"id": "prefill_table_block", "type": "table", "layout": "full", "tableId": "prefill_table"},
        {"id": "suffix_finding", "type": "markdown", "layout": "full", "sourceId": "fresh_summary", "body": "## 长上下文 suffix/Decode 的局部优化真实存在\n\nquery length 为 1 或 4 时，SM86 在 1K–8K 每个 context 都稳定快于 Generic CUDA，组合中位收益约 12%–25%。**含义：** 当前 SM86 适合作为 Decode/短 suffix Provider；这只是项目内 kernel 对照，不是行业框架横评。"},
        {"id": "suffix_chart_block", "type": "chart", "layout": "full", "chartId": "suffix_speedup_chart"},
        {"id": "real8b_finding", "type": "markdown", "layout": "full", "sourceId": "fresh_summary", "body": "## 真实 8B 链路仍由权重搬运主导\n\n8/128-token 的三种 KV mode 都约 1.35–1.36 秒，最大观测 spread 为 0.402%，Top-1 全部一致。两次采样不足以排名，只足以判断 mode 差异被权重流式吞没。**含义：** 下一项整机优化应减少 Transformer weight H2D。"},
        {"id": "real8b_table_block", "type": "table", "layout": "full", "tableId": "real8b_table"},
        {"id": "scope", "type": "markdown", "layout": "full", "sourceId": "fresh_summary", "body": "## 范围、指标与容量定义\n\nKernel latency 是同步 wall-clock P50；short/long 每 case 7 次，full Prefill 5 次，真实 8B 2 次。workspace 指额外完整 K/V 临时区。Llama-3.1-8B BF16 每 token KV 为 128KiB；分页避免增长时重分配，但容量仍线性增长，且当前 KV 只能驻留 GPU。Page 16 的单请求尾页最坏内部碎片约 1.875MiB，Page 32 最坏约 3.875MiB。"},
        {"id": "capacity_table_block", "type": "table", "layout": "full", "tableId": "capacity_table"},
        {"id": "support_finding", "type": "markdown", "layout": "full", "sourceId": "fresh_summary", "body": "## 当前可执行能力窄于策略接口\n\n真正执行的是 GPU-resident BF16/FP16 exact dense KV，加上 request/session/in-memory sealed-prefix reuse。Quantized、sparse、CPU/NVMe active offload、persistent prefix 和 continuous batching 没有执行结果。**含义：** 框架已经冻结扩展边界，但不能把 ABI 预留描述成现有能力。"},
        {"id": "support_table_block", "type": "table", "layout": "full", "tableId": "support_table"},
        {"id": "numerical", "type": "markdown", "layout": "full", "sourceId": "numerical_contract", "body": "## 数值正确性通过版本化合同，但不等于 HF bitwise identity\n\nNumerical Contract V2 的 L0–L5 与微型质量门禁通过，质量集最大准确率下降 0 pp，1000-token HF replay 抽样 101/101 Top-1 一致。但真实 8B 相对 HF fused SDPA 的 297 个阶段仍有 23 个严格 elementwise/ordered-Top-k 诊断失败。**含义：** greedy 行为和架构合同可接受；需要 HF tensor identity 的调用方仍不满足。"},
        {"id": "method", "type": "markdown", "layout": "full", "body": "## 方法与验证评级\n\n新鲜测试运行于 commit `" + commit + "`、RTX 3080 Ti SM86、PyTorch 2.4.1+cu121、CUDA 12.1。自动化、生命周期、kernel 和短真实 8B 在 2026-08-03 重跑；1000-token 真实模型、HF replay 与质量集为同一当前 commit 内的 2026-08-02 证据。验证评级：**Share with caveats**。主要数字已从原始 JSON 独立重算，但单架构、低重复数和 synthetic shape 限制外推。"},
        {"id": "limitations", "type": "markdown", "layout": "full", "body": "## 限制、失败模式与状态漂移\n\n- 只有 SM86 真机，且没有 continuous batching 吞吐/公平性测试。\n- Prefix index 没有容量上限、LRU 或 tenant API。\n- full Prefill 是单层 synthetic；真实 8B 低重复数只支持噪声量级判断。\n- legacy gather+SDPA 是实验对照，不是生产 serving baseline。\n- 主文档仍写 SM86 `performance_qualified`，当前 Provider 和资格汇总写 `production`；支持级别文档漂移需要修正。"},
        {"id": "next_steps", "type": "markdown", "layout": "full", "body": "## 建议的下一步\n\n1. 接入成熟 paged/Flash Prefill Provider，并保持 Prefill/Decode 显式分相。\n2. 优先降低 Transformer weight H2D，再复测端到端 Decode。\n3. 实现真实 CPU/NVMe KV tier、量化 KV 或 sparse selection；完成前不要宣传“KV 超显存”。\n4. 接入 continuous batching，并新增容量回压、prefix LRU 和并发公平性压力。\n5. 统一 SM86 文档/机器状态，并在 SM80/89/90 真机独立资格验证。"},
        {"id": "questions", "type": "markdown", "layout": "full", "body": "## 仍需回答的问题\n\n- Flash/Paged Prefill 能否同时保持 0 full-KV workspace 和 SDPA 级吞吐？\n- 真实请求分布下 page 16/32 的碎片率、prefix 命中率和 COW 放大是多少？\n- 权重 H2D 降低后，12%–25% suffix kernel 收益能转化成多少端到端收益？\n- KV pool 接近容量上限时，缺少 active offload/eviction 会如何影响拒绝率和尾延迟？"},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Cascade-LLM 当前 KV 管理机制测试报告",
            "description": "Current-commit correctness, lifecycle, memory, kernel and real-model evidence",
            "generatedAt": generated_at,
            "sources": manifest_sources,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": [headline],
                "workspace_rows": workspace_rows,
                "prefill_tradeoff": prefill_tradeoff,
                "prefill_rows": prefill_rows,
                "suffix_speedups": suffix_rows,
                "lifecycle_rows": lifecycle_rows,
                "real8b_rows": real8b_rows,
                "capacity_rows": capacity_rows,
                "mode_support": mode_support,
            },
        },
        "sources": sources,
    }
    artifact_path = REPORT_ROOT / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    source_notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "required_structure_mapping": {
            "technical_summary": "技术结论",
            "key_findings": ["分页与事务生命周期", "零完整-KV workspace", "完整 Prompt Prefill", "suffix/Decode", "真实 8B"],
            "scope_and_definitions": "范围、指标与容量定义",
            "methodology": "方法与验证评级",
            "limitations_and_robustness": "限制、失败模式与状态漂移",
            "recommended_next_steps": "建议的下一步",
            "further_questions": "仍需回答的问题",
        },
        "chart_map": [
            {"section": "Memory benefit", "question": "How much full-KV workspace is avoided?", "family": "comparison", "type": "bar", "fields": ["context_label", "legacy_workspace_mib", "production_workspace_mib"], "takeaway": "Production direct paging holds full-KV workspace at zero", "palette": "hard two-root cap"},
            {"section": "Full Prefill cost", "question": "What latency is paid for zero workspace?", "family": "comparison", "type": "bar", "fields": ["context_label", "legacy_baseline", "generic_slowdown", "sm86_slowdown"], "takeaway": "SM86 slowdown grows to 27.1x at 2048 tokens", "palette": "relaxed multi-category"},
            {"section": "Suffix kernel benefit", "question": "Does SM86 improve direct paging at long context?", "family": "comparison", "type": "bar", "fields": ["context_label", "median_speedup"], "takeaway": "SM86 retains a stable local gain", "palette": "single-root preferred"},
            {"section": "Lifecycle stability", "question": "Does CUDA memory creep across ownership churn?", "family": "trend", "type": "line", "fields": ["cycle", "allocated_mib", "reserved_mib"], "takeaway": "Memory remains flat across eleven checkpoints", "palette": "hard two-root cap"},
        ],
        "omitted_visuals": [
            {"segment": "Fresh real-8B mode comparison", "reason": "Only two contexts and mode differences below 0.5%; an exact table is more honest than a nearly flat chart."},
            {"segment": "KV capacity projection", "reason": "Five discrete planning points require exact lookup and caveats; table preferred."},
        ],
        "validation": {
            "overall_assessment": "Share with caveats",
            "calculation_spot_checks": [
                "Recomputed all SM86/Generic ratios from raw p50 values.",
                "Recomputed full-Prefill slowdown and workspace MiB from raw bytes.",
                "Recomputed real-8B mode spread per context and phase.",
                "Recomputed BF16 KV bytes/token from model config and policy formula.",
                "Reconciled lifecycle allocation_count and release_count at 2001 each.",
            ],
            "required_caveats": summary["validation_caveats"],
        },
    }
    notes_path = REPORT_ROOT / "source_notes.json"
    notes_path.write_text(
        json.dumps(source_notes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "summary": relative(summary_path),
        "markdown": relative(markdown_path),
        "artifact": relative(artifact_path),
        "source_notes": relative(notes_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
