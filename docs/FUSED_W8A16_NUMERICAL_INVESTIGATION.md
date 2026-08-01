# Fused W8A16 数值差异调查

> 日期：2026-08-01
> 范围：Llama 3.1 8B INT8 symmetric per-channel、BF16 activation/scale、SM86、CUTLASS provider ABI 2。
> 对照：相同 INT8 checkpoint，prefill 均使用 `int8_dequant_bf16_fallback`，只在 decode 比较 fallback 与 `fused_w8a16`。

## 1. 结论

原报告中的 5 个失败 stage 不是 5 个独立 kernel 缺陷。首个超过原 stage 级 `torch.allclose(atol=0.08, rtol=0.04)` 门槛的是：

```text
decode_0 / layer 31 / MLP output
```

随后误差传播到 decode 0 的 final norm/logits，以及 decode 1 的 final norm/logits。逐 linear 使用完全相同的 activation、INT8 weight 和 scale 重算后，448/448 个真实 decode linear 均在原 elementwise bound 内，没有单个 linear 失败。

根因是 fused GEMV 与 PyTorch BF16 `F.linear` 的累加与舍入语义不同：

1. fused M=1 kernel 先将 `INT8 × BF16 scale` 舍入为 BF16 weight，以 FP32 累加，最后舍入为 BF16 output；
2. PyTorch fallback 先生成相同 BF16 weight，但 cuBLAS BF16 GEMM 默认允许 reduced-precision reduction；
3. 对 K=14336 的 down projection，默认 reduced-precision reduction 是主要差异来源；
4. 对 K=4096 的矩阵，剩余差异主要来自 cuBLAS 与自定义 GEMV 的归约顺序；
5. 很小的逐 linear 差异经过 32 层残差、RMSNorm、SwiGLU 和 attention 传播，最终使 5 个 stage 越过原本不适合作为 fused 端到端门禁的 elementwise `allclose`。

placement 已通过 resident/streamed 297/297 bitwise 一致验证，不是根因。

## 2. 五个失败 stage

固定输入：

```text
prefill: 128000,128006,882,220
decode:  128001,128001
```

| Stage | 具体位置 | max abs | mean abs | 传播判断 |
|---|---|---:|---:|---|
| `decode_0/mlp/31` | Layer 31 MLP，gate/up/down 之后 | 0.171875 | 0.010256 | 首个越过旧 stage 门槛 |
| `decode_0/final_norm` | Layer 31 hidden 之后 | 0.265625 | 0.054610 | MLP/残差误差经 RMSNorm 放大 |
| `decode_0/logits` | BF16 LM Head | 0.343750 | 0.057568 | final norm 误差传播；Top-1 一致，Top-10 交集 90% |
| `decode_1/final_norm` | 第二个 decode 的 Layer 31 之后 | 0.343750 | 0.055549 | 前一步 KV/hidden 差异继续传播 |
| `decode_1/logits` | BF16 LM Head | 0.375000 | 0.054491 | Top-1 一致，Top-10 集合 100% 一致但顺序不同 |

Layer 31 的三个 MLP linear 在相同输入下均没有越界：

| Matrix | input range | scale range | local max abs | local mean abs | local mean relative |
|---|---:|---:|---:|---:|---:|
| gate_proj | -2.0781 ～ 3.4219 | 0.000278 ～ 0.006073 | 0.03125 | 0.000629 | 0.0828% |
| up_proj | -2.0781 ～ 3.4219 | 0.000236 ～ 0.005280 | 0.015625 | 0.000491 | 0.0838% |
| down_proj | -3.9375 ～ 13.125 | 0.000372 ～ 0.004303 | 0.0078125 | 0.000759 | 0.2113% |

Layer 31 down projection 的同输入 local mean error 只占该层完整 MLP stage mean error 的约 7.4%，其余主要来自前 31 层已经积累的 activation 差异。hidden mean error 从 layer 0 的约 0.000151，逐层增长到 layer 31 的约 0.015412。

## 3. 已排除项目

真实三组 golden 共检查 1344 个 linear：

- activation/output 为 BF16；INT8 storage、BF16 scale 和 BF16 compute dtype 全部一致；
- 1344/1344 的 K 均满足 16 对齐，N 均满足 8 对齐；
- logical shape 与 storage shape 完全一致，没有 padding；
- per-channel scale shape 均为 `[N, 1]`；
- resident 与 streamed raw tensor 布局的结果 bitwise 一致；
- 当前“先解量化舍入、再 GEMV”比“INT8 GEMM 后乘 scale”更接近 fallback。

三组 golden 的平均统计：

```text
fused vs fallback mean(abs error)          ≈ 9.41e-5（基线 prompt）
fused vs FP32 accumulation mean(abs error) ≈ 4.94e-8
fallback vs FP32 mean(abs error)           ≈ 9.41e-5
post-scale vs fallback mean(abs error)      ≈ 8.21e-4
```

因此不能把 scale 移回 GEMM 之后；该变体误差约大一个数量级。

## 4. BF16 reduction 开关证据

固定随机真实 shape micro-test：

```text
hidden_hidden: M=1, N=4096, K=4096
down:          M=1, N=4096, K=14336
```

K=14336 时：

| 对比 | max abs | mean abs | mean relative |
|---|---:|---:|---:|
| fused vs 默认 fallback | 1.0 | 0.046393 | 0.2090% |
| fused vs 禁用 reduced-precision fallback | 0.0625 | 0.0000153 | 0.000069% |
| fused vs FP32 accumulation | 0.0625 | 0.0000153 | 0.000069% |
| 禁用 reduced-precision fallback vs FP32 | 0 | 0 | 0 |

关闭 `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` 显著减少 K=14336 的局部误差，但不能让完整模型 bitwise 一致：K=4096 上仍有不同归约顺序，并会经 32 层传播。因此不把修改全局 PyTorch 开关作为 runtime 修复，也不改变生产 fallback 的既有语义。

## 5. Golden contract v1

contract：

```text
tests/fixtures/fused_w8a16_sm86_golden_v1.json
```

证据集：3 个固定真实 prompt × 2 个 decode step，共 1344 次逐 linear 对比。观测包络：

```text
local max abs                         0.125
local max mean-relative              0.23134%
fused-vs-FP32 max mean-relative       0.001255%
hidden max/mean                       0.25 / 0.01610
final norm max/mean                   0.50 / 0.05555
logits max/mean                       0.375 / 0.05757
logits Top-1                          6/6 一致
logits Top-10 intersection            最低 90%
```

v1 门槛从三组观测包络向上取整，包含明确余量：

- 每个 local linear 仍必须满足原 `atol=0.08, rtol=0.04` elementwise bound，允许越界元素数为 0；
- local mean-relative ≤ 0.3%；
- fused 相对 FP32 累加 mean-relative ≤ 0.002%；
- attention max/mean ≤ 0.08/0.005；
- MLP max/mean ≤ 0.25/0.015；
- hidden max/mean ≤ 0.375/0.025；
- final norm max/mean ≤ 0.625/0.07；
- logits max/mean ≤ 0.5/0.075；
- logits Top-1 必须一致，Top-10 集合交集率不得低于 90%。

这些门槛不是对原 allclose 的临时放宽：它同时增加了逐 linear 同输入门禁、FP32 累加门禁、各类 stage 的聚合门禁和任务输出 Top-k 门禁。修改门槛必须创建新 contract 版本并附带新的固定 case 证据。

## 6. 回归命令

一键运行全部 golden：

```bash
.venv/bin/python tools/run_fused_golden.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B-Instruct-W8A16 \
  --output-dir real_results/8b_w8a16/golden_v1
```

检查已有单份诊断：

```bash
.venv/bin/python tools/check_fused_diagnostic.py \
  --report real_results/8b_w8a16/golden_v1/baseline_special_tokens.json \
  --contract tests/fixtures/fused_w8a16_sm86_golden_v1.json
```

任一 local、FP32、stage 或 Top-k 门禁失败时命令返回非零退出码。
