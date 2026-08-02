# Cascade-LLM KV Mode 性能与生产门禁报告

## 技术结论

本轮已经关闭原有两个生产门禁：L4 的真实 8B Provider 非退化微型质量集通过；L5 的 2 MiB reserved 漂移被证明为 PyTorch caching allocator 保留段，1000-cycle 期间 reserved 恒定，关闭后显式 trim 回到零。SM86 Kernel 相对 Generic CUDA 的全部 BF16/FP16 case 中位加速为 **1.129×**，长上下文收益稳定；但真实 8B Decode 仍约 **1358.9 ms/token**，Attention 优化基本被每 Token Transformer 权重传输掩盖。

更重要的负面结果是：真实 8B、2048-token Prefill 中，SM86 TTFT 为 **9337.1 ms**，而实验 gather+SDPA 为 **4919.5 ms**。SM86 production path 慢 **1.90×**。因此当前 SM86 可以通过既定生产稳定性和数值合同，但长 Prefill Kernel 仍有明显性能缺陷，不应描述为全面优于成熟 SDPA。

## 两个生产门禁已经关闭

L4 覆盖 64 个困惑度 Token、8 个短文本题、4 个 499–2413 Token 长检索题和 6 个固定对话题。SM86 相对显式 gather+SDPA reference 的困惑度变化为 **-0.0875%**，三类准确率下降均为 **0 pp**，候选与 reference 均答对全部 18 道选择题。

L5 的 1000-cycle 测试执行 1000 次 Beam Fork/Discard、1000 次 Speculative Fork/Commit、142 次 Session Fork 和 62 次 Prefix 注册/命中。page/ref/pin 全部归零；运行期间 CUDA reserved span 为 0。关闭后 trim 前缓存池保留 2 MiB，trim 后 reserved drift 为 0，因此它不是 KV Runtime 泄漏。

## Kernel 模式：SM86 长上下文稳定优于 Generic

Kernel microbenchmark 使用 HND、GQA 32/8 heads、head dim 128。短/中矩阵为 batch 2、Prefill query 8；长矩阵为 batch 1、Prefill query 4。每 case 2 次 warmup、7 次正式运行。SM86/Generic 全部 production case 的速度比范围为 **0.995×–1.751×**，中位 **1.129×**。少量短 case 的最低值低于 1% 属于计时噪声范围；长上下文所有 case 均明确快于 Generic。

Page 16 与 Page 32 的长上下文中位延迟比为 **1.036**，没有足够证据宣布固定页大小普遍更快；应根据上下文和 workload 选择。BF16/FP16 中位延迟比为 **1.002**，在 SM86 上也没有形成决定性差异。

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
