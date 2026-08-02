# KV Framework V1 Numerical Contract V2 生产资格报告

**复验日期：** 2026-08-02

**分支：** `framework-v0`

**模型：** Llama-3.1-8B-Instruct BF16

**实机：** NVIDIA GeForce RTX 3080 Ti 12GB（SM86）

**环境：** PyTorch 2.4.1+cu121、CUDA Runtime 12.1

## 结论

```text
SM86 / BF16 / HND / Paged ABI 1: production
KV Framework V1 exact GPU path:      production qualified
SM80 / SM89 / SM90:                  declared / unqualified
```

原有两个阻塞项已经关闭：L4 真实 8B Provider 非退化质量微型集通过；L5
ownership 1000-cycle 中观察到的 2 MiB 是 PyTorch CUDA caching allocator 的空闲保留段，
不是存活 Tensor、Page 或引用泄漏。关闭后显式 trim 使 allocated/reserved 均回到基线。

本结论只适用于上述 SM86、Provider ABI、模型、dtype 和布局组合，不代表其他 GPU、
量化 KV、Sparse KV 或 CPU/NVMe 活跃 Offload 已取得生产资格。

机器可读结论：

```text
real_results/kv_v2/kv_qualification_summary.json
real_results/kv_v2/kv_numerical_summary.json
real_results/kv_v2/kv_ownership_summary.json
real_results/kv_v2/kv_performance_summary.json
real_results/kv_v2/compatibility_matrix.json
real_results/kv_v2/manifest.json
```

完整自动化回归为 `148/148` 通过。

## L0–L5 结果

| Level | 结果 | 核心证据 |
|---|---|---|
| L0 算子安全 | 通过 | MHA/GQA/MQA × Prefill/Decode 6/6；shape、dtype、finite、page index 全通过 |
| L1 局部算子 | 通过 | 显式 FP32 math attention 为真值；candidate 误差在原生 BF16 baseline 的 2× envelope 内 |
| L2 模型阶段 | 通过 | attention、residual、MLP、final norm、logits 均落入独立版本化 envelope |
| L3 Logits/生成 | 通过 | Golden Top-1 3/3；Top-10 集合门禁通过；1000-token 抽样 101/101 Top-1 |
| L4 模型质量 | 通过 | PPL 64 tokens；短题 8、长检索 4、对话 6；准确率下降 0 pp |
| L5 生产稳定 | 通过 | 真实 8B 1000 token 零漂移；ownership 1000-cycle page/ref/pin/线程与 trim 后 CUDA 漂移均为 0 |

## L4：真实模型质量非退化

质量集用于比较同一 BF16 checkpoint 的 SM86 direct-paged Provider 与显式
`legacy_gather_sdpa_reference`，不是通用模型能力榜单。

| 指标 | Reference | SM86 | 变化/门禁 |
|---|---:|---:|---:|
| Perplexity（64 teacher-forced tokens） | 28.587486 | 28.562472 | -0.0875%；退化记为 0 |
| 短文本选择题 | 8/8 | 8/8 | 下降 0 pp |
| 长上下文检索（499–2413 prompt tokens） | 4/4 | 4/4 | 下降 0 pp |
| 固定对话题 | 6/6 | 6/6 | 下降 0 pp |

所有 logits 和 NLL 均 finite。Contract 同时强制最低覆盖量，减少样本数不能通过门禁。

## L5：2 MiB reserved 的根因与修复

原报告在 Runtime 关闭后看到：

```text
allocated drift: 0 bytes
reserved drift:  2,097,152 bytes
```

逐 cycle 采样确认 reserved 不增长，运行期跨度为 0；Page Pool、ref_count、pin_count、
线程数也保持稳定。根因是 PyTorch caching allocator 将已经释放的约 1.3 MiB GPU Store
所在的 2 MiB segment 留在缓存池。测试现区分两类信号：

1. 运行期 `allocated` 必须精确回到基线，`reserved` 不得逐步增长；
2. close 后记录 trim 前缓存量，再调用 `torch.cuda.empty_cache()`，trim 后
   allocated/reserved drift 必须均为 0。

复验执行了 1000 次 Beam Fork/Discard、1000 次 Speculative Fork/Commit、142 次
Session Fork、62 次 Prefix 注册/命中，最终 Page/ref/pin 全部归零，trim 后漂移 0。

## 23 个旧严格阶段失败

旧 HF-SDPA identity 门禁的 23/297 个失败仍保留为诊断证据，没有删除或放宽成临时
容差。V2 的归因是：

- 每个失败都出现在该 phase 首次 attention 非零差异之后；
- resident 与 streamed production 路径 297/297 bitwise 一致，排除 placement；
- legacy gather + 相同 HF-SDPA 恢复 297/297，排除 Page mapping、COW 和 LM Head；
- HF SDPA 与 HF eager 控制实验也在 19/23 相同阶段超过旧门限；
- 根因是 direct-paged FP32 reduction 与 fused SDPA reduction tree 的舍入传播；
- 所有阶段均处于独立 SM86/provider/ABI/model Numerical Contract V2 envelope。

## 性能资格与已知缺陷

Kernel 矩阵覆盖 BF16/FP16、Page 16/32、Prefill/Decode、16/128/512 和
4K/8K/32K 上下文。SM86 相对 Generic CUDA 的全部 case 中位加速为 1.129×，范围
0.995×–1.751×；长上下文 case 全部获得明确正收益。Production Provider 的完整 KV
workspace 和完整 score matrix 均为 0。

真实 8B 单 Token Decode 约 1358.9 ms/token，三种 Attention mode 的差异被每 Token
Transformer 权重流式传输掩盖。另一个必须公开的缺陷是：2048-token 完整 Prefill 中，
SM86 TTFT 为 9337.1 ms，legacy gather+SDPA 为 4919.5 ms，SM86 慢 1.90×。这不破坏
数值、稳定性和内存资格，但说明当前 Prefill Kernel 不是成熟 SDPA/Flash Kernel 的
全面性能替代。

详细结果见 `docs/KV_MODE_PERFORMANCE_REPORT.md` 和便携 HTML 报告。

## Runtime / Provider 边界

页面生命周期仍完全属于 Runtime：

```text
PagedKVRuntime
├── PagePool / RequestTable
├── OwnershipManager       ref/pin/Fork/COW
├── PrefixCache
├── PagedAttentionBackend  prefill/decode/workspace only
└── PagedKVKernelBackend   append/copy payload only
```

Provider 不能 allocate、release、retain、pin、unpin 或执行 Fork/COW。无 GPU Provider
时 PagePool/Fork/COW 单元测试仍可通过；替换 Attention 或 Page Kernel Backend 不改变
Block Table 与生命周期语义。Production 路径没有隐藏 gather/SDPA fallback。

## 硬件矩阵

| 架构 | 状态 | 结论 |
|---|---|---|
| SM80 | `declared` | Bundle/Capability 可加载；无 A100 真机资格证据 |
| SM86 | `production` | RTX 3080 Ti L0–L5、真实模型、性能与 148 项回归通过 |
| SM89 | `declared` | Bundle/Capability 可加载；无 RTX 4090/L4 真机资格证据 |
| SM90 | `declared` | Bundle/Capability 可加载；无 H100/H200 真机资格证据 |

## 未完善部分

1. 2048-token 及更长完整 Prefill 的 SM86 Kernel 性能明显落后成熟 SDPA；应接入
   paged Flash/FlashInfer 类 Provider，并保持 Prefill/Decode backend 显式配置。
2. L4 当前是小型 Provider 非退化集，仍需补充公开 PPL、理解和长上下文数据集。
3. 10000-token 扩展 soak 尚未执行；当前生产门禁为 1000 token。
4. SM80、SM89、SM90 缺少真机 L0–L5，不能继承 SM86 Numerical Contract。
5. Prefix Cache 容量/LRU/tenant API、量化 KV、CPU/NVMe 活跃 Offload、Quest 仍为
   experimental 或 unsupported。

最终标识：

```text
KV Framework V1 / SM86 exact BF16 HND ABI1 / production qualified
```
