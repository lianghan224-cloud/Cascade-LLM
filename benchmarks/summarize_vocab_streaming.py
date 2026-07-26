#!/usr/bin/env python3
"""Summarize the real Llama-3.1-8B vocabulary-streaming experiment."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "real_results"
BASELINE_PATH = (
    RESULTS
    / "airllm"
    / "bench_cascade_full_pinned_matrix_s2_gpu0_rerun.json"
)
GROUP_RESIDENT_PATH = (
    RESULTS / "bench_full_pinned_matrix_group_vocab_resident_s2.json"
)
STREAMED_PATH = (
    RESULTS / "bench_full_pinned_matrix_group_vocab_streamed_s2.json"
)
CORRECTNESS_PATH = (
    RESULTS
    / "correctness_comparison_vocab_streamed_matrix_group_8token.json"
)
SUMMARY_PATH = RESULTS / "vocab_streaming_summary.json"
REPORT_PATH = RESULTS / "vocab_streaming_report.md"
GIB = 1024**3
MIB = 1024**2


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    baseline = load(BASELINE_PATH)
    group_resident = load(GROUP_RESIDENT_PATH)
    streamed = load(STREAMED_PATH)
    correctness = load(CORRECTNESS_PATH)

    baseline_ms = baseline["decode_wall_distribution_ms"]["median"]
    resident_ms = group_resident["decode_wall_distribution_ms"]["median"]
    streamed_ms = streamed["decode_wall_distribution_ms"]["median"]
    baseline_peak = baseline["cuda_peak_allocated_bytes"]
    resident_peak = group_resident["cuda_peak_allocated_bytes"]
    streamed_peak = streamed["cuda_peak_allocated_bytes"]
    transformer_profile = streamed["profiles"][1]
    vocab_profile = streamed["vocab_profiles"][1]

    summary = {
        "schema_version": 1,
        "scope": {
            "model": "Meta-Llama-3.1-8B BF16",
            "gpu": "NVIDIA GeForce RTX 3080 Ti",
            "prompt_tokens": streamed["prompt_tokens"],
            "warmup_decode": streamed["warmup_decode"],
            "decode_repeats": streamed["decode_repeats"],
            "weight_store": streamed["weight_store"],
            "transformer_granularity": streamed["granularity"],
            "vocab_mode": streamed["vocab_mode"],
            "top_k": streamed["top_k"],
        },
        "layout": {
            "vocab_size": 128256,
            "hidden_size": 4096,
            "embedding_and_lm_head_tied": False,
            "shared_cpu_layout_and_scheduler": True,
            "lm_head_chunk_bytes": 128 * MIB,
            "lm_head_chunk_rows": vocab_profile["chunk_rows"],
            "lm_head_chunk_count": vocab_profile["chunk_count"],
            "last_chunk_rows": 13568,
            "transformer_groups_per_layer": 4,
            "transformer_transfer_units": streamed["runtime"][
                "transfer_units"
            ],
        },
        "results": {
            "previous_matrix_resident": {
                "decode_median_ms": baseline_ms,
                "tokens_per_second": 1000.0 / baseline_ms,
                "cuda_peak_allocated_bytes": baseline_peak,
            },
            "matrix_group_resident_vocab": {
                "decode_median_ms": resident_ms,
                "tokens_per_second": 1000.0 / resident_ms,
                "cuda_peak_allocated_bytes": resident_peak,
            },
            "matrix_group_streamed_vocab": {
                "decode_median_ms": streamed_ms,
                "decode_p10_ms": streamed[
                    "decode_wall_distribution_ms"
                ]["p10"],
                "decode_p90_ms": streamed[
                    "decode_wall_distribution_ms"
                ]["p90"],
                "tokens_per_second": 1000.0 / streamed_ms,
                "prefill_ms": streamed["prefill_wall_ms"],
                "cuda_peak_allocated_bytes": streamed_peak,
                "cuda_peak_reserved_bytes": streamed[
                    "cuda_peak_reserved_bytes"
                ],
                "planned_weight_gpu_bytes": streamed["runtime"][
                    "weight_gpu_bytes"
                ],
            },
        },
        "ratios": {
            "latency_increase_vs_previous_pct": (
                streamed_ms / baseline_ms - 1.0
            )
            * 100.0,
            "throughput_retained_vs_previous_pct": (
                baseline_ms / streamed_ms
            )
            * 100.0,
            "peak_reduction_vs_previous_pct": (
                1.0 - streamed_peak / baseline_peak
            )
            * 100.0,
            "previous_peak_over_streamed_peak": (
                baseline_peak / streamed_peak
            ),
            "peak_bytes_saved_vs_previous": baseline_peak - streamed_peak,
            "vocab_streaming_latency_increase_vs_group_resident_pct": (
                streamed_ms / resident_ms - 1.0
            )
            * 100.0,
            "matrix_group_latency_change_vs_previous_pct": (
                resident_ms / baseline_ms - 1.0
            )
            * 100.0,
        },
        "profiles": {
            "transformer": transformer_profile,
            "vocab": vocab_profile,
            "total_streamed_weight_bytes": (
                transformer_profile["h2d_bytes"]
                + vocab_profile["h2d_bytes"]
            ),
            "embedding_decode_bytes": 8192,
        },
        "correctness": correctness,
        "source_files": [
            str(BASELINE_PATH.relative_to(ROOT)),
            str(GROUP_RESIDENT_PATH.relative_to(ROOT)),
            str(STREAMED_PATH.relative_to(ROOT)),
            str(CORRECTNESS_PATH.relative_to(ROOT)),
        ],
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    ratios = summary["ratios"]
    result = summary["results"]["matrix_group_streamed_vocab"]
    report = f"""# Llama-3.1-8B CPU词表流式化真实实验

## 实现范围

- Transformer每层分为QKV、O、Gate/Up、Down四个完整矩阵组；
- 不切单个矩阵内部Tile，不使用自定义算子；
- Embedding根据Token ID从CPU `[V,H]` 行连续布局提取所需行；
- LM Head按词表行切成8个约128 MiB块，每块16,384行，末块13,568行；
- 两个词表矩阵共享CPU布局和调度实现，但Llama-3.1-8B
  `tie_word_embeddings=false`，数值权重保持独立；
- LM Head使用标准`F.linear`和`torch.topk`在线归并全局Top-10。

## 结果

| 配置 | 中位ms/token | token/s | CUDA peak allocated |
|---|---:|---:|---:|
| 上一版本：matrix、词表常驻 | {baseline_ms:.3f} | {1000 / baseline_ms:.3f} | {baseline_peak / GIB:.3f} GiB |
| matrix_group、词表常驻 | {resident_ms:.3f} | {1000 / resident_ms:.3f} | {resident_peak / GIB:.3f} GiB |
| matrix_group、词表流式 | {streamed_ms:.3f} | {1000 / streamed_ms:.3f} | {streamed_peak / GIB:.3f} GiB |

相对上一版本，词表流式版本：

- CUDA peak allocated降低 **{ratios["previous_peak_over_streamed_peak"]:.2f}倍**，
  节省 **{ratios["peak_reduction_vs_previous_pct"]:.2f}%**、
  {(ratios["peak_bytes_saved_vs_previous"] / GIB):.3f} GiB；
- 延迟增加 **{ratios["latency_increase_vs_previous_pct"]:.2f}%**，
  保留 **{ratios["throughput_retained_vs_previous_pct"]:.2f}%** 吞吐；
- P10–P90为
  {result["decode_p10_ms"]:.3f}–{result["decode_p90_ms"]:.3f} ms；
- 计划GPU权重区为
  {result["planned_weight_gpu_bytes"] / MIB:.3f} MiB，实测峰值为
  {streamed_peak / MIB:.3f} MiB。

矩阵分组本身相对上一版本的延迟变化只有
{ratios["matrix_group_latency_change_vs_previous_pct"]:.3f}%，新增延迟主要来自
LM Head流式传输。

## LM Head诊断

每Token新增传输{vocab_profile["h2d_bytes"] / 1e9:.3f} GB：

- 8次H2D累计{vocab_profile["h2d_event_sum_ms"]:.3f} ms；
- H2D有效吞吐{vocab_profile["h2d_effective_gbps"]:.3f} GB/s；
- 分块GEMV和在线Top-k累计
  {vocab_profile["compute_event_sum_ms"]:.3f} ms；
- LM Head流水总计{vocab_profile["wall_ms"]:.3f} ms。

Embedding在单Token decode只传输8 KiB，不再完整加载约0.979 GiB矩阵。

## 正确性

连续8个greedy token全部匹配CPU参考。完整`[8,128256]`词表logits最低
cosine为{min(correctness["cosine_similarity_per_step"]):.6f}，8步argmax
全部匹配；误差与上一版BF16专用执行器处于同一范围。在线Top-k由每块局部
Top-k和全局候选再次Top-k构成，不使用近似词表裁剪。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote {}".format(REPORT_PATH))


if __name__ == "__main__":
    main()
