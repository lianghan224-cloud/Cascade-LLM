# M6 真实性能收敛验证

> 验证日期：2026-08-01
> 模型：Llama 3.1 8B Instruct
> GPU：NVIDIA GeForce RTX 3080 Ti 12 GB（SM86）
> 结论边界：本文数据是本机单请求、batch=1 的短序列工程验证，不代表服务吞吐。

## 1. 本轮实现

- benchmark report schema v2 增加每次 forward 的 Transformer/LM Head H2D、有效 host copy/H2D 带宽、source/compute stall、overlap、resident/streamed bytes 和 hit ratio。
- `tools/benchmark_m6a.py` 固化 M6A 的 3 种传输粒度、3 种 prefill 长度、2 种 decode 长度、2 次 warmup 和 5 次正式运行，并支持中断后续跑。
- `StaticTransformerPlacement` 作为 ExecutionPlan schema v1 之外的 runtime sidecar，按完整层前缀选择静态常驻权重；冻结接口没有改动。
- `MixedResidentDeviceArena` 和 `MixedRuntime` 支持常驻层直接计算、剩余层继续经过有界双流流水线；常驻层和流式层使用同一个显式 backend。
- Embedding 与 LM Head placement 可以独立选择 `auto|resident|streamed`。
- `MemoryPlanner` 在 GPU 分配前预算 Transformer resident arena，并保留 KV、workspace、slot 和 CUDA reserve 的优先级。
- `tools/quantize_llama_checkpoint.py` 可将真实 BF16 Llama checkpoint 的 Transformer linear 转为 INT8 symmetric per-channel，保留 BF16 Norm、Embedding 和 LM Head。
- `tools/compare_cascade_paths.py` 支持显式 prefill/decode backend 以及两侧不同 resident budget，用于隔离 provider 数值差异和验证 placement 不改变结果。
- benchmark 记录 decode P50/P95/P99、首末窗口延迟和长期漂移比例。

## 2. 真实 checkpoint

BF16：

```text
/ssd/cascade-llm/models/Llama-3.1-8B-Instruct
```

本轮生成的 W8A16：

```text
/ssd/cascade-llm/models/Llama-3.1-8B-Instruct-W8A16
```

量化结果：

- 224 个 Transformer projection matrix 转为 INT8 per-channel；
- checkpoint 大小 9,083,953,152 bytes；
- scale 为 BF16，Norm/Embedding/LM Head 为 BF16；
- checkpoint manifest 全量校验通过；
- 权重重建最大绝对误差 0.004150390625，加权平均绝对误差 0.00011247。

## 3. BF16 常驻曲线

以下均为 8-token prefill、2-token decode、matrix_group、2 slots、pinned staging、streamed LM Head 的短跑中位数。它用于验证实现和趋势，不替代完整 M6A 矩阵。

| Transformer 预算 | 常驻层比例 | Transformer H2D/forward | Decode ms/token | GPU peak allocated |
|---:|---:|---:|---:|---:|
| 0 GiB | 0% | 13.000 GiB | 1457.7 | 0.449 GiB |
| 2 GiB | 12.5% | 11.375 GiB | 1292.1 | 2.074 GiB |
| 4 GiB | 28.1% | 9.344 GiB | 1081.9 | 4.105 GiB |
| 8 GiB | 59.4% | 5.281 GiB | 674.6 | 8.168 GiB |
| 10 GiB | 75.0% | 3.250 GiB | 471.4 | 10.199 GiB |

10 GiB resident 配置与 Hugging Face 参考逐阶段比较 297/297 通过；resident 层不再产生 H2D，H2D bytes 随 streamed 权重近似线性下降。

LM Head 独立 placement 的结果：

| Transformer 预算 | LM Head | Decode ms/token | LM Head H2D/forward | GPU peak allocated |
|---:|---|---:|---:|---:|
| 0 GiB | streamed | 1457.7 | 0.9785 GiB | 0.449 GiB |
| 0 GiB | resident | 1354.1 | 0 | 1.435 GiB |
| 8 GiB | streamed | 674.6 | 0.9785 GiB | 8.168 GiB |
| 8 GiB | resident | 578.3 | 0 | 9.154 GiB |

## 4. W8A16 性能曲线

同样使用短序列工程配置。`fused_w8a16` 是显式选择的 CUTLASS provider；报告中没有 fallback。

| 模式 | Transformer resident | LM Head | Transformer H2D/forward | Decode ms/token | GPU peak allocated |
|---|---:|---|---:|---:|---:|
| INT8 fallback 全流式 | 0% | streamed | 6.500 GiB | 798.9 | 0.667 GiB |
| fused W8A16 全流式 | 0% | streamed | 6.500 GiB | 799.6 | 0.230 GiB |
| fused W8A16 4 GiB | 59.4% | streamed | 2.640 GiB | 415.1 | 4.091 GiB |
| fused W8A16 8 GiB | 100% | streamed | 0 | 131.9 | 6.733 GiB |
| fused W8A16 8 GiB | 100% | resident | 0 | 46.27 | 7.720 GiB |

结论：全流式时瓶颈仍是 pageable→pinned/source wait/H2D，所以 fused 与 fallback 的端到端延迟基本相同；fused 的价值在于不申请完整 BF16 解量化 workspace，并允许 12 GB GPU 常驻完整 INT8 Transformer。完整常驻 Transformer 和 LM Head 后，本机 decode 从 BF16 全流式约 1458 ms/token 降到约 46 ms/token。

全流式与 8 GiB Transformer resident 的同一 fused 路径逐阶段比较为 297/297 bitwise 一致，证明 placement 不改变计算结果。

## 5. 1000-token 稳定性

配置：真实 W8A16、fused W8A16、8 GiB Transformer budget、resident LM Head、8-token prefill、1000-token decode。

| 指标 | 结果 |
|---|---:|
| 平均 decode | 46.22 ms/token |
| P50 / P95 / P99 | 46.20 / 46.44 / 46.92 ms |
| 首 100 token 平均 | 46.06 ms |
| 末 100 token 平均 | 46.36 ms |
| 延迟漂移 | +0.66% |
| Transformer H2D/forward | 0 |
| LM Head H2D/forward | 0 |
| GPU peak allocated/reserved | 7.86 / 7.89 GiB |

请求正常完成并释放 KV handle；没有 queue、worker、CUDA allocator 或 event 错误。

## 6. 数值状态和未完成验收

INT8 量化质量（INT8 fallback 对原始 BF16）：固定短输入的 prefill/decode Top-1 均一致，Top-10 集合交集率为 90%～100%；logits 平均绝对误差约 0.116～0.134。量化模型不应按 BF16 的逐层严格一致阈值验收。

CUTLASS provider 与同一 INT8 权重的显式 BF16 解量化 reference 仍存在舍入顺序差异：

- M=1 已改为先按 BF16/FP16 舍入解量化权重，再做 GEMV；单矩阵资格测试通过已有 provider 容差；
- 显式 fallback prefill + fused decode 的 prefill 完全一致；两个 decode 的 Top-1 一致，Top-10 交集率 90%/100%；
- 仍有 5/297 个逐阶段项目未通过当前 `atol=0.08, rtol=0.04 + Top-k 顺序完全一致` 规则；
- M>1 per-channel CUTLASS 路径仍是整数 GEMM 后按列应用 scale，与完整 BF16 weight 解量化的舍入顺序不同。

因此本轮可以确认真实 fused 性能路径可运行、可常驻、无静默 fallback，但不能宣称它已经与 fallback 逐层严格一致。不能通过放宽阈值或删除失败 stage 来关闭该问题。下一步应选择：实现更接近 reference 舍入语义的 epilogue/mainloop，或正式定义并冻结 fused provider 独立的端到端数值容差与质量门槛。

后续 P0 调查已完成根因定位并冻结独立 golden contract v1：差异主要来自 PyTorch BF16 reduced-precision reduction 和 cuBLAS/GEMV 归约顺序，三个真实固定 prompt 的 1344 次逐 linear 回归均通过。旧的 5 个失败 stage 仍保留作为传播证据。详见 `docs/FUSED_W8A16_NUMERICAL_INVESTIGATION.md`。

完整 M6A 的 8/128/512 prefill × 32/128 decode × 3 granularity × 5 repeats 仍未执行；当前只完成短跑排序（layer > matrix > matrix_group）和稳定性验证。任何最终粒度结论必须以完整矩阵为准。

现有 Hugging Face 约 38.45 ms/token 的结果使用两张 GPU 常驻 BF16 权重；本轮约 46 ms/token 使用单张 GPU、INT8 Transformer 和 resident BF16 LM Head。两者精度、GPU 数量和 placement 不同，只能分别作为可用性边界，不能放入同精度同硬件排名。llama.cpp/vLLM 的 M6G 公平基准尚未开始。

## 7. 回归结果

```text
Ran 78 tests in 22.191s
OK
```

冻结的 ExecutionPlan schema v1 和 M5 API fixture 均通过。主要报告保存在：

```text
real_results/8b_bf16/
real_results/8b_w8a16/
```

下一优先级：

1. 执行完整 M6A 长矩阵并生成可续跑汇总；
2. 为 fused W8A16 冻结可解释的端到端质量门槛，或修正 M>1/M=1 舍入语义；
3. 在 8 GiB 常驻模式下做 5 次正式长 decode，而不是用单次 1000-token 替代方差统计；
4. 再进入 pinned staging worker/NUMA 优化与外部框架公平基线。
