# KV Numerical Contract V2

V2 将显式 FP32 math attention 设为算子主参考，HF-SDPA、FlashAttention 和
cuDNN SDPA 只作为横向 baseline。合同按 architecture、provider、ABI、model、dtype
和 layout 分开版本化。

## L0–L5

- L0：shape、dtype、finite、page/block 边界和无非法访问；
- L1：候选相对 FP32 的 max/mean/P99 absolute error 与 cosine loss，不超过原生
  BF16/FP16 baseline 相对同一 FP32 参考的 2 倍；max relative error 记录但不作为
  近零值门禁；
- L2：attention、residual、MLP、final norm、logits 进入版本化模型 envelope，且没有
  未解释的层间突增；
- L3：固定 prompt Top-1 100%，Top-10 集合至少 90%，长序列 teacher-forced Top-1
  至少 99.9%；当前合同还要求至少 100 个采样位置；
- L4：PPL 相对退化不超过 0.1%，短理解、长检索、固定对话下降均不超过
  0.2 个百分点；
- L5：至少 1000 token、allocator/page/ref/pin/workspace 零漂移、1000-cycle
  ownership 与延迟趋势门禁。

只有 L0–L5 和性能门禁全部通过才可使用 `production`。

## Envelope 管理

当前 SM86 BF16 合同位于：

```text
tests/fixtures/kv_numerical_sm86_bf16_abi1_v2.json
```

Envelope 不能在测试失败后直接调宽。更新必须：

1. 生成相同输入的 FP32、native BF16/FP16、candidate 三方证据；
2. 标明首次差异 layer、attention/MLP、矩阵、输入和 scale 范围；
3. 证明没有 layout、padding、越界或 placement 错误；
4. 提升合同版本或创建新的完整 key；
5. 评审质量集与长稳结果后再合并。

BF16 合同不得复用于 INT8/FP8/INT4 KV。

## 复现

```bash
.venv/bin/python tools/qualify_paged_hardware.py \
  --device cuda:0 \
  --output /results/hardware_provider_v2.json

.venv/bin/python tools/qualify_kv_v2.py \
  --contract tests/fixtures/kv_numerical_sm86_bf16_abi1_v2.json \
  --hardware /results/hardware_provider_v2.json \
  --model-comparison /results/model_comparison.json \
  --attribution /results/attribution.json \
  --long-replay /results/long_replay.json \
  --soak /results/soak.json \
  --ownership /results/ownership.json \
  --automated-tests /results/test_summary.json \
  --performance /results/performance_short.json /results/performance_long.json \
  --quality /results/quality.json \
  --output-dir /results/kv_v2
```

省略 `--quality` 会显式生成 `L4: not_run`，不会静默跳过。
