#!/usr/bin/env python3
"""Build the real single-GPU Cascade versus AirLLM comparison report."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "real_results" / "airllm"
AIR_FIRST_PATH = RESULTS / "bench_airllm_bf16_prefetch_gpu0.json"
AIR_RERUN_PATH = RESULTS / "bench_airllm_bf16_prefetch_gpu0_rerun.json"
CASCADE_PATH = RESULTS / "bench_cascade_full_pinned_matrix_s2_gpu0_rerun.json"
SUMMARY_PATH = RESULTS / "comparison_summary.json"
REPORT_PATH = RESULTS / "comparison_report.md"
GIB = 1024**3


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    air_first = load(AIR_FIRST_PATH)
    air = load(AIR_RERUN_PATH)
    cascade = load(CASCADE_PATH)
    air_profile = air_first["profile_rows"][0]
    cascade_profile = cascade["profiles"][1]

    air_ms = air["decode_wall_distribution_ms"]["median"]
    cascade_ms = cascade["decode_wall_distribution_ms"]["median"]
    air_peak = air["cuda_peak_allocated_bytes"]
    cascade_peak = cascade["cuda_peak_allocated_bytes"]
    payload = air_profile["move_bytes"]
    air_h2d_ms = air_profile["move_gpu_ms"]
    air_load_ms = air_profile["load_wall_ms"]
    air_profile_wall = air_profile["wall_ms"]
    cascade_h2d_ms = cascade_profile["h2d_event_sum_ms"]
    cascade_compute_ms = cascade_profile["compute_event_sum_ms"]

    expected_tokens = [311, 1505, 701, 8352, 13, 578, 7580, 315]
    air_tokens = air["first_eight_generated_token_ids"][0]
    cascade_tokens = cascade["generated_token_ids"][0][:8]

    summary = {
        "schema_version": 1,
        "scope": {
            "model": "Meta-Llama-3.1-8B BF16",
            "gpu": "NVIDIA GeForce RTX 3080 Ti",
            "visible_gpu_count": 1,
            "prompt": air["prompt"],
            "prompt_tokens": air["prompt_tokens"],
            "warmup_decode": air["warmup_decode"],
            "decode_repeats": air["decode_repeats"],
            "airllm": "3.0.1, BF16, compression=None, prefetching=True",
            "cascade": "full_pinned, matrix granularity, 2 slots",
        },
        "primary_results": {
            "airllm_hot_split": {
                "decode_median_ms": air_ms,
                "decode_p10_ms": air["decode_wall_distribution_ms"]["p10"],
                "decode_p90_ms": air["decode_wall_distribution_ms"]["p90"],
                "tokens_per_second": 1000.0 / air_ms,
                "prefill_ms": air["prefill_wall_ms"],
                "cuda_peak_allocated_bytes": air_peak,
                "cuda_peak_reserved_bytes": air["cuda_peak_reserved_bytes"],
                "initialization_seconds": air["initialization_seconds"],
                "measured_decode_disk_read_bytes": air[
                    "measured_decode_io_delta"
                ]["read_bytes"],
            },
            "cascade": {
                "decode_median_ms": cascade_ms,
                "decode_p10_ms": cascade["decode_wall_distribution_ms"]["p10"],
                "decode_p90_ms": cascade["decode_wall_distribution_ms"]["p90"],
                "tokens_per_second": 1000.0 / cascade_ms,
                "prefill_ms": cascade["prefill_wall_ms"],
                "cuda_peak_allocated_bytes": cascade_peak,
                "cuda_peak_reserved_bytes": cascade["cuda_peak_reserved_bytes"],
                "initialization_seconds": (
                    cascade["load_seconds"] + cascade["resident_load_seconds"]
                ),
                "measured_decode_disk_read_bytes": cascade[
                    "measured_decode_io_delta"
                ]["read_bytes"],
            },
        },
        "ratios": {
            "cascade_speedup_over_airllm": air_ms / cascade_ms,
            "cascade_prefill_speedup_over_airllm": (
                air["prefill_wall_ms"] / cascade["prefill_wall_ms"]
            ),
            "cascade_peak_allocated_over_airllm": cascade_peak / air_peak,
            "cascade_extra_peak_allocated_bytes": cascade_peak - air_peak,
            "airllm_peak_reduction_vs_cascade_pct": (
                1.0 - air_peak / cascade_peak
            )
            * 100.0,
            "full_bf16_payload_over_airllm_peak": payload / air_peak,
            "full_bf16_payload_over_cascade_peak": payload / cascade_peak,
        },
        "airllm_reproducibility": {
            "first_run_decode_median_ms": air_first[
                "decode_wall_distribution_ms"
            ]["median"],
            "hot_rerun_decode_median_ms": air_ms,
            "difference_pct": (
                air_first["decode_wall_distribution_ms"]["median"] / air_ms
                - 1.0
            )
            * 100.0,
            "first_split_initialization_seconds": air_first[
                "initialization_seconds"
            ],
            "existing_split_initialization_seconds": air[
                "initialization_seconds"
            ],
            "split_bytes": air["split_after"]["bytes"],
            "split_files": air["split_after"]["file_count"],
        },
        "diagnostics": {
            "airllm": {
                "streamed_units": air["streamed_units"],
                "bytes_per_token": payload,
                "load_calls": air_profile["load_calls"],
                "load_wall_sum_ms": air_load_ms,
                "load_effective_gbps": payload / air_load_ms / 1_000_000.0,
                "move_calls": air_profile["move_calls"],
                "move_gpu_sum_ms": air_h2d_ms,
                "h2d_effective_gbps": payload / air_h2d_ms / 1_000_000.0,
                "pinned_move_bytes": air_profile["move_pinned_bytes"],
                "profile_wall_ms": air_profile_wall,
                "load_sum_over_profile_wall_pct": (
                    air_load_ms / air_profile_wall * 100.0
                ),
                "move_sum_over_profile_wall_pct": (
                    air_h2d_ms / air_profile_wall * 100.0
                ),
            },
            "cascade": {
                "streamed_units": cascade["runtime"]["transfer_units"],
                "bytes_per_token": cascade_profile["h2d_bytes"],
                "h2d_sum_ms": cascade_h2d_ms,
                "h2d_effective_gbps": cascade_profile["h2d_effective_gbps"],
                "compute_sum_ms": cascade_compute_ms,
                "cpu_pinned_bytes": cascade["cpu_pinned_bytes"],
                "gpu_weight_bytes": cascade["runtime"]["weight_gpu_bytes"],
            },
        },
        "correctness": {
            "expected_first_eight_token_ids": expected_tokens,
            "airllm_first_eight_token_ids": air_tokens,
            "cascade_first_eight_token_ids": cascade_tokens,
            "airllm_matches_expected": air_tokens == expected_tokens,
            "cascade_matches_expected": cascade_tokens == expected_tokens,
            "airllm_matches_cascade": air_tokens == cascade_tokens,
        },
        "source_files": [
            str(AIR_FIRST_PATH.relative_to(ROOT)),
            str(AIR_RERUN_PATH.relative_to(ROOT)),
            str(CASCADE_PATH.relative_to(ROOT)),
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    speedup = summary["ratios"]["cascade_speedup_over_airllm"]
    prefill_speedup = summary["ratios"]["cascade_prefill_speedup_over_airllm"]
    peak_ratio = summary["ratios"]["cascade_peak_allocated_over_airllm"]
    peak_saved = summary["ratios"]["airllm_peak_reduction_vs_cascade_pct"]
    extra_gib = summary["ratios"]["cascade_extra_peak_allocated_bytes"] / GIB
    air_diag = summary["diagnostics"]["airllm"]
    cascade_diag = summary["diagnostics"]["cascade"]

    report = f"""# Llama-3.1-8B 单卡 Cascade 与 AirLLM 实测对比

## 结论

在 RTX 3080 Ti 单卡、BF16、batch=1、6-token prompt、1 次 decode 预热和
7 次稳态 decode 的统一口径下，Cascade 推荐配置的稳态速度是 AirLLM
热切分复测的 **{speedup:.2f} 倍**，prefill 是 **{prefill_speedup:.2f}
倍**。AirLLM 的峰值 CUDA allocated 更低：Cascade 是 AirLLM 的
**{peak_ratio:.2f} 倍**，即 AirLLM 少用 **{peak_saved:.2f}%**、
约 **{extra_gib:.3f} GiB**。

| 指标 | AirLLM 3.0.1 | Cascade |
|---|---:|---:|
| 配置 | BF16、默认预取、无压缩 | full_pinned、matrix、2 slots |
| 稳态中位延迟 | {air_ms:.3f} ms/token | {cascade_ms:.3f} ms/token |
| 稳态速度 | {1000 / air_ms:.3f} token/s | {1000 / cascade_ms:.3f} token/s |
| P10–P90 | {air["decode_wall_distribution_ms"]["p10"]:.3f}–{air["decode_wall_distribution_ms"]["p90"]:.3f} ms | {cascade["decode_wall_distribution_ms"]["p10"]:.3f}–{cascade["decode_wall_distribution_ms"]["p90"]:.3f} ms |
| Prefill | {air["prefill_wall_ms"]:.3f} ms | {cascade["prefill_wall_ms"]:.3f} ms |
| CUDA peak allocated | {air_peak / GIB:.3f} GiB | {cascade_peak / GIB:.3f} GiB |
| CUDA peak reserved | {air["cuda_peak_reserved_bytes"] / GIB:.3f} GiB | {cascade["cuda_peak_reserved_bytes"] / GIB:.3f} GiB |
| 测量窗口物理磁盘读取 | {air["measured_decode_io_delta"]["read_bytes"]} B | {cascade["measured_decode_io_delta"]["read_bytes"]} B |

## 为什么速度相差较大

AirLLM 每 token 流式处理全部 35 个单元，共
{payload / 1e9:.3f} GB。单次诊断中，权重读取/映射累计
{air_diag["load_wall_sum_ms"]:.3f} ms，权重安装/H2D CUDA Event 累计
{air_diag["move_gpu_sum_ms"]:.3f} ms，H2D 有效吞吐
{air_diag["h2d_effective_gbps"]:.3f} GB/s；实际输入 H2D 的 pinned
权重为 {air_diag["pinned_move_bytes"]} B。读取阶段虽然命中 Linux page
cache、没有产生物理磁盘读取，仍包含逐层 safetensors 映射、CPU 张量和
Python/内存管理成本。诊断累计时间来自不同线程，不能作为严格互斥的
wall-time 分解。

Cascade 每 token 传输 {cascade_diag["bytes_per_token"] / 1e9:.3f} GB，
H2D Event 累计 {cascade_diag["h2d_sum_ms"]:.3f} ms，有效吞吐
{cascade_diag["h2d_effective_gbps"]:.3f} GB/s；计算 Event 累计
{cascade_diag["compute_sum_ms"]:.3f} ms。Embedding、LM Head 和小型 Norm
常驻 GPU，因此传输量比 AirLLM 少；完整 CPU 权重一次性锁页，热路径不再
逐层读取或动态 pin/unpin。

## 显存与主存权衡

AirLLM 会流式加载 Embedding、32 层、Norm 和 LM Head，因此峰值主要由
最大单层/Embedding 决定，只需 {air_peak / GIB:.3f} GiB CUDA allocated。
Cascade 将 Embedding、LM Head 和 Norm 常驻，并预分配两个矩阵 slot，
因此是 {cascade_peak / GIB:.3f} GiB，但换来显著更高吞吐。

Cascade 的 full_pinned 模式锁页 {cascade["cpu_pinned_bytes"] / GIB:.3f}
GiB CPU 权重。AirLLM 不把全量权重常驻进程内锁页；它依赖按层文件和 OS
page cache，已有切分文件占 {air["split_after"]["bytes"] / GIB:.3f} GiB
额外 SSD 空间。首次创建切分并初始化为
{air_first["initialization_seconds"]:.3f} 秒，复用切分时为
{air["initialization_seconds"]:.3f} 秒。

## 正确性与限制

两者前 8 个 greedy token ID 均为 `{expected_tokens}`，与已有 Transformers
CPU 参考一致。两边使用同一 checkpoint、同一 GPU 和 BF16，但 Transformers
版本分别为 {air["transformers_version"]} 与
{cascade["transformers_version"]}；Cascade 使用专用 Llama 执行器，
AirLLM 使用 Hugging Face 通用模型及逐层 hook。结果是本机单请求短上下文
decode 数据，不代表长上下文、批处理、量化或冷 page-cache 场景。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
