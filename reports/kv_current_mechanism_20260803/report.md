# Cascade-LLM 当前 KV 管理机制测试报告

## 技术结论

当前 KV Framework V1 是一个**生命周期可靠、显存边界清晰的 exact dense GPU paged KV 基础设施**，适合单请求 Decode、短 Prefill 和后续调度器/新 Provider 接入；它还不是高吞吐长 Prompt 或超显存 KV 的完整解决方案。

- **优点可复现：** 148/148 自动化测试通过；1000 轮 Fork/COW/Prefix 压力后 page/ref/pin、CUDA 内存和线程漂移均为 0；全部 64 个 production benchmark case 不创建完整 KV workspace。
- **Kernel 局部有效：** SM86 对 Generic CUDA 的短中矩阵中位加速为 1.0846×，1K–8K suffix/decode 矩阵中位加速为 1.1563×。
- **完整 Prefill 是核心短板：** 2048-token full Prefill 中 SM86 比 gather+SDPA 慢 27.096×；它省下每层 32MiB workspace，但牺牲了成熟 SDPA 的吞吐。
- **端到端收益尚未显现：** 真实 Llama-3.1-8B 的短上下文三模式最大观测差异仅 0.402%，Decode 约 1.35 秒/token，权重流式 H2D 掩盖了 Attention kernel 优化。
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

新鲜 short matrix 中 SM86 相对 `reference_paged_exact` 的最大 pairwise absolute error 为 0.0009765625。当前 Numerical Contract V2 的 L0–L5 与微型质量门禁通过，质量集的最大准确率下降为 0 pp，1000-token HF replay 抽样 101/101 Top-1 一致。

同时，真实 8B 相对 HF fused SDPA 的 297 个阶段中仍有 23 个严格 elementwise/ordered-Top-k 诊断失败。当前资格合同允许 architecture-specific reduction envelope，所以状态汇总为 production；需要 strict HF tensor identity 的调用方仍应视为不满足。

## 方法与复现口径

新鲜测试运行于 git commit `a614b485370500ceda84662572e60759835a835d`、RTX 3080 Ti SM86、PyTorch 2.4.1+cu121、CUDA 12.1。自动化、生命周期、kernel 和短真实 8B 均在 2026-08-03 重新执行。1000-token 真实模型长稳、101-position HF replay 和 L4 微型质量集来自同一当前 commit 内已签入的 2026-08-02 证据。

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
