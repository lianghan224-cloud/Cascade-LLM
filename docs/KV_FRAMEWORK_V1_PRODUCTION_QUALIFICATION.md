# KV Framework V1 Numerical Contract V2 资格报告

**复验日期：** 2026-08-02

**分支：** `framework-v0`

**模型：** Llama-3.1-8B-Instruct BF16

**实机：** NVIDIA GeForce RTX 3080 Ti 12GB（SM86）

**环境：** PyTorch 2.4.1+cu121、CUDA Runtime 12.1

## 结论

```text
SM86 / BF16 / HND / Paged ABI 1: performance_qualified
KV Framework V1:                    非 production
```

SM86 已通过 L0–L3、真实 8B、1000-token 主请求稳定性和代表性性能门禁。
它不能标记为 `production`，原因是：

1. L4 模型质量数据集尚未运行；
2. L5 的 Fork/COW/Prefix 1000-cycle 报告存在 2 MiB CUDA reserved drift，
   不符合 V2 的严格零漂移要求；
3. SM80、SM89、SM90 没有对应真机，保持 `declared`。

机器可读结论位于：

```text
real_results/kv_v2/kv_qualification_summary.json
real_results/kv_v2/kv_numerical_summary.json
real_results/kv_v2/compatibility_matrix.json
real_results/kv_v2/manifest.json
```

本轮完整自动化回归为 `146/146` 通过（含 CPU 与当前 SM86 CUDA 路径）。

## L0–L5 结果

| Level | 结果 | 核心证据 |
|---|---|---|
| L0 算子安全 | 通过 | MHA/GQA/MQA × prefill/decode 6/6；shape/dtype/finite/page index 全通过 |
| L1 局部算子 | 通过 | 显式 FP32 math attention 为真值；候选 max/mean/P99/cosine loss 均不超过原生 BF16 baseline 的 2× |
| L2 模型阶段 | 通过 | attention/residual/MLP/final norm/logits 全部落入 SM86+ABI1+8B V2 envelope，无未解释突增 |
| L3 Logits/生成 | 通过 | 固定位置 Top-1 3/3、Top-10 集合最低 90%、1000-token 路径抽样 101/101 Top-1 |
| L4 模型质量 | **未运行** | 缺少 PPL、短理解、长检索、固定对话四类数据集 |
| L5 生产稳定 | **未通过** | 真实 8B 主请求 1000 token 零漂移；ownership 1000-cycle reserved drift 为 2 MiB |

L1 的 FP32 参考由显式 QK、FP32 softmax 和 V matmul 构成，不调用 HF-SDPA、
FlashAttention 或 cuDNN SDPA。上述实现只作为横向 baseline，不能单独充当真值。

## 23 个旧严格阶段失败

旧 HF-SDPA identity 门禁的 23/297 个失败没有删除，也没有通过修改 atol/rtol
掩盖。V2 将它们保留为诊断证据，而不再要求不同 reduction tree 逐元素一致。

完整归因结论如下：

- 每个失败都位于该 phase 的首次 attention 非零差异之后；
- resident 与 streamed production 路径 297/297 bitwise 一致，排除 placement；
- legacy gather + 同一 HF-SDPA 可恢复 297/297，说明 Page mapping、COW 和 LM Head
  不是根因；
- HF SDPA 与 HF eager 控制实验也在 19/23 相同阶段超过旧门限；
- 根因是 direct-paged FP32 reduction 与私有 fused SDPA reduction tree 的舍入传播；
- 没有 unexplained spike，所有阶段均进入独立 SM86/provider/ABI/model envelope。

L2 的当前观察值：

| 阶段 | count | max abs | 最大 mean abs | V2 上限（max / mean） |
|---|---:|---:|---:|---:|
| attention | 96 | 0.125 | 0.003974 | 0.16 / 0.005 |
| residual | 96 | 0.460938 | 0.023033 | 0.60 / 0.030 |
| MLP | 96 | 0.375 | 0.017044 | 0.50 / 0.025 |
| final norm | 3 | 0.921875 | 0.046898 | 1.00 / 0.060 |
| logits | 3 | 0.515625 | 0.070655 | 0.70 / 0.080 |

这些 envelope 只适用于 `SM86 / sm86 provider / ABI 1 / Llama-3.1-8B /
BF16 / HND`。架构、ABI、dtype、layout 或模型变化必须建立新合同。

## 1000-token 与 ownership 长稳

真实 8B Greedy 自回归：

```text
tokens:                              1000
mean / p50:           1459.61 / 1458.87 ms/token
last / first quartile:            0.99944
CUDA allocated drift:                    0
CUDA reserved drift:                     0
pages/ref_count/pin_count leak:           0
workspace peak:                           0
provider fallback:                     null
```

Fork/COW/Prefix/Beam/Speculative 1000-cycle 的 ownership graph、ref_count、
pin_count、页面释放和线程数量都通过，但 CUDA reserved 增加 2 MiB。它可能是延迟
初始化或 allocator 保留；在增加 pre-warm snapshot 并重新得到严格 0 之前，L5 必须失败。

## 性能与消融

短/中上下文（16/128/512，page 16/32，decode 与 suffix prefill）共 36 个 case；
长上下文（4K/8K/32K）共 24 个 case。结果：

- production provider workspace 全部为 0；
- production 相对 Python `reference_paged_exact` 的最低加速为 53.50×；
- SM86 相对 Generic：短/中为 1.01×–1.75×，长上下文为 1.12×–1.20×；
- 没有完整连续 K/V gather、完整 score matrix 或 SDPA 隐藏 fallback。

这是当前 correctness-first Provider 内部比较，不代表优于 FlashInfer、vLLM 或
FlashAttention；完整 8B 单 token 延迟仍主要受 streamed weight H2D 限制。

## Runtime/Provider 边界与瘦身

当前结构：

```text
PagedKVRuntime（493 行协调器）
├── RequestTable            request → block table
├── OwnershipManager        ref/pin/Fork/COW/event
├── PrefixCache             prefix 索引与额外引用
├── KVExecutionCoordinator  kernel 输入和执行编排，无所有权
├── PagedAttentionBackend   prefill/decode/workspace
└── PagedKVKernelBackend    append/copy page payload
```

Provider Bundle 不能暴露 allocate/release/retain/pin/unpin/Fork。替换 Attention 或
Page Kernel Backend 不改变 Block Table 和生命周期语义。旧 `kv_cache.py`、online
attention 和连续 SDPA 消融已迁入 `layer_streaming/experimental/`，默认导入图不加载；
旧 `PagedAttentionProvider` 及各架构 Provider 别名已删除。

## 硬件矩阵

| 架构 | 当前状态 | 结论 |
|---|---|---|
| SM80 | `declared` | Bundle/Capability 路径存在；无 A100 真机证据 |
| SM86 | `performance_qualified` | RTX 3080 Ti L0–L3、真实模型与性能通过；非 production |
| SM89 | `declared` | Bundle/Capability 路径存在；无 RTX 4090/L4 真机证据 |
| SM90 | `declared` | Bundle/Capability 路径存在；无 H100/H200 真机证据 |

不得把 `declared` 或在 SM86 上成功导入 Python 类描述成目标架构兼容。

## 仍需完成

1. 运行并固化 L4 四类质量集；
2. 对 ownership soak 增加显式 CUDA warmup 后重新验证 reserved drift 为 0；
3. 执行 10000-token 扩展 soak，并记录完整 P50/P95/P99；
4. 在 SM80、SM89、SM90 真机分别运行 L0–L5；
5. 将 SM86 correctness-first kernel 替换或优化为 tiled production kernel；
6. 完成 Prefix Cache 容量/LRU/tenant API；量化 KV、CPU/NVMe、Quest 仍为
   experimental/unsupported。

在这些门禁通过前，最终标识保持：

```text
KV Framework V1 RC / SM86 performance_qualified / not production
```
