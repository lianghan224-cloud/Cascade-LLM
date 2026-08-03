# Cascade-LLM Codex 开发交接

- 更新时间：2026-08-03
- 交接分支：`framework-v0`
- 交接前基线提交：`a614b485370500ceda84662572e60759835a835d`

## 1. 一句话目标

构建一个**模型完整权重显著超过 GPU 显存仍能正确运行**的单机 Llama-family 推理框架：权重驻留 CPU，按有界粒度流入 GPU，用固定资源池、异步 H2D/Compute 重叠、显式 KV 和内存规划实现可预测运行。

本项目当前不是通用推理服务器，也不追求先覆盖所有模型或并发场景。第一原则是让超显存模型真正跑起来，并把正确性、显存、数据搬运和性能瓶颈测清楚。

## 2. 当前开发决策

KV 管理整改已达到 **KV Stack Beta**，现阶段暂时搁置。新的主线是迁移到更大服务器，开展更大权重实验。除非大权重实验直接暴露 KV 阻塞问题，否则不要继续扩大 KV 范围。

优先实验问题：

1. 目标 checkpoint 能否在 GPU 分配前通过 config/index/key/shape/dtype 校验？
2. CPU RAM、pinned staging、GPU slot、常驻词表、激活和 KV 的预算是否闭合？
3. BF16/FP16 权重流式路径能否完成短 Prefill + Decode，Top-1/数值是否可信？
4. 实际瓶颈在 pageable→pinned、H2D、GEMM、Attention、LM Head 还是同步/队列？
5. 增加 slot、prefetch depth、granularity 或静态常驻预算后，端到端是否真实改善？

## 3. 已实现的系统

### 3.1 超显存权重执行主线

- 从 Hugging Face `config.json` 构建 Llama 几何与执行计划，不固定模型尺寸。
- safetensors 单文件、多 shard、index、alias/tied weight、shape/dtype/offset 启动前校验。
- `full_pinned`：完整权重 arena 锁页；RAM/memlock 要求高。
- `pinned_staging`：完整权重放 pageable CPU arena，小型 pinned slot 做 H2D staging；大模型默认从这里开始。
- matrix、matrix_group、layer 三种传输粒度。
- BF16/FP16 dense，INT8/INT4 显式解量化 fallback。
- SM86 CUTLASS W8A16 可选 Provider；支持范围必须按 phase/shape/quant layout 资格检查。
- Transformer 权重流式、Embedding/LM Head resident 或 streamed、LM Head 分块 Top-k。
- Copy Stream、Compute Stream、可复用 CUDA Event、1..4 slots 和有界 prefetch 队列。
- MemoryPlanner 在大内存分配前预算 RAM、memlock、GPU slot、resident 权重、workspace、KV、Embedding、LM Head 和安全余量。
- JSON RunReport、阶段 profile、资源稳定性与 benchmark schema。

主要入口：

- `tools/run_llama31.py`
- `tools/benchmark.py`
- `tools/compare_reference.py`
- `tools/chat_llama31_70b_int8.py`
- `layer_streaming/memory.py`
- `layer_streaming/mixed_runtime.py`
- `layer_streaming/multi_dtype_store.py`

### 3.2 KV 当前状态

已完成：

- generation-safe PageHandle、HND Page Arena、请求 LogicalBlockTable。
- PagePool/Ownership 单一生命周期权威，ref/pin/compute/IO 配对。
- Append 事务、Fork、尾页 COW、Prefix、Beam/branch、Speculative commit、跨页 rollback。
- Dense 和可运行的 Quest-flat CPU summary index。
- Mock GPU/CPU/SSD TieredKVStore、迁移状态机、checksum、authority、prefetch 去重/取消和 FIFO Coordinator。
- Full Prefill、Chunked Prefill、Decode、Short Suffix 显式分类。
- SM86 Decode/Short Suffix 直接 paged CUDA；Full/Chunked Prefill 当前进入 `reference_paged_exact` correctness fallback。

验证结果：

- 统一 full profile：69 PASS、8 SKIPPED_WITH_REASON、0 FAIL、0 BLOCKED。
- 152 个仓库单元测试通过。
- 100,002 次随机生命周期操作，全部页面归还。
- 50-loop CUDA allocated/reserved drift 都为 0。
- 当前简单 Decode 微基准中，SM86 相对 legacy gather 约 2.65–2.82 倍，完整 KV workspace 为 0。

权威文档：

- `docs/KV_ARCHITECTURE_CONTRACT.md`
- `docs/KV_CURRENT_STATE_AUDIT.md`
- `docs/KV_REMEDIATION_IMPLEMENTATION_REPORT.md`
- `reports/kv_validation.md`
- `reports/kv_validation.json`

不要使用旧文档中“Quest/Prefix/非连续 Page 未实现”之类的历史结论覆盖以上状态。

## 4. 尚未完成或不能宣称

- 没有专用高吞吐 Full/Chunked Prefill Kernel；当前 correctness fallback 会显著拖慢长 Prompt。
- 真实 GPU↔CPU↔NVMe/GDS KV 没接入活动 PagedKVRuntime；TieredKVStore 是状态正确的 Mock。
- Quest 没有真实模型/数据集质量资格。
- 只有 RTX 3080 Ti / SM86 完成当前完整合成资格；SM80/SM89/SM90 不得继承该结论。
- Llama Executor 仍以单请求 adapter 为主，未形成服务级 continuous batching。
- 没有真实 34B/70B 当前版本的完整正确性、长生成与公平性能资格。
- INT8/INT4 大部分路径仍是显式 fallback；不能宣传为成熟 Tensor Core 推理。
- 权重流式 8B 历史端到端 Decode 约 1.36 s/token，主要受权重搬运支配；Kernel 局部收益不能直接替代系统收益。

## 5. 大权重实验路线

### P0：新服务器 Bring-up

1. 按 `docs/SERVER_MIGRATION.md` clone，不复制旧 `.venv` 或本机 `.so`。
2. 运行 `tools/capture_environment.py`，保存目标硬件基线。
3. 重建 Python 3.10 + 锁定依赖，运行 `scripts/verify_server.sh`。
4. 根据 Compute Capability 选择 Provider。非 SM86 先用 `generic_cuda`；不要强行加载 SM86 binary。

完成标准：环境报告、doctor、全单元测试和 GPU 基础检查通过，Git 工作树保持可解释。

### P1：Checkpoint 和内存规划

1. 使用 `scripts/download_codellama34b_bf16.sh --dry-run` 或同类受控下载/局域网 rsync。
2. 固定模型仓库、revision、文件 manifest；模型不进 Git。
3. 先执行 metadata/checkpoint validation 和 MemoryPlanner，不直接启动完整推理。
4. 校验 RAM ≥ checkpoint arena + 系统余量；默认 `pinned_staging`，确认 memlock 后才考虑 `full_pinned`。
5. KV `max_cache_length` 只覆盖实际 prompt + decode，不使用模型声明的最大上下文作为默认预分配。

完成标准：无大权重 GPU 分配前即可获得明确的 checkpoint 与内存可行性报告。

### P2：最小真实推理

建议第一条真实命令采用：

- batch 1；
- 8–32 token prompt；
- 1–4 decode token；
- `pinned_staging`；
- `matrix_group`；
- 1 或 2 slots；
- streamed Embedding/LM Head（显存紧张时）；
- Dense GPU KV、page size 16、紧凑 max cache。

先保存 Top-1、logits/hidden 对比和完整 RunReport，再增加上下文与 token 数。

完成标准：至少一次完整 Prefill+Decode 正常退出，CPU/pinned/GPU/KV/线程资源释放闭合，并有可复现命令。

### P3：瓶颈归因

按以下顺序实验，每次只改变一个变量：

1. slots 1→2→3；
2. prefetch depth；
3. matrix_group vs layer；
4. Embedding/LM Head resident vs streamed；
5. 静态 GPU resident weight budget；
6. pinned_staging vs full_pinned（仅资源允许）；
7. BF16 vs已资格的量化路径。

报告必须分开：pageable→pinned、H2D、dequant、GEMM、Attention、LM Head、同步等待、TTFT、decode ms/token 和峰值内存。

### P4：根据证据优化

优先级由真实大模型报告决定，候选方向包括：

- 减少 H2D 总字节或提高权重复用。
- NUMA 绑定、pinned arena 和源数据准备并行。
- 更适合大矩阵的传输粒度和 slot 调度。
- 词表流式/常驻取舍。
- 目标 GPU 架构 Provider 资格和 Kernel 调优。

只有长 Prompt 成为实际主线阻塞时，才恢复专用 paged Prefill Kernel；真实 Tiered KV、Quest 和服务调度继续后置。

## 6. 实验最小记录模板

每次实验至少保存：

```text
git_commit:
checkpoint_repo/revision:
checkpoint_manifest:
gpu / compute_capability / driver:
python / torch / torch_cuda:
weight_format / backend / provider:
weight_store / granularity / slots / prefetch_depth:
embedding_mode / lm_head_mode / resident_budget:
kv_backend / page_size / max_cache_length:
prompt_tokens / decode_tokens / batch:
peak_ram / pinned / gpu / kv:
ttft_ms / decode_ms_per_token:
h2d_ms / compute_ms / attention_ms / lm_head_ms / waits:
correctness_result:
exit_status / failure_reason:
command:
```

## 7. 新 Codex 会话建议首条指令

在仓库根目录启动 Codex 后使用：

```text
完整读取 AGENTS.md、docs/CODEX_HANDOFF.md 和 docs/SERVER_MIGRATION.md。
当前主线是模型完整权重远超显存的大权重推理实验，KV Stack Beta 暂停扩展。
先运行只读环境审计和 scripts/verify_server.sh，核对目标 GPU 架构、依赖、checkpoint 与内存预算；
不要直接开始长生成，不要使用 SM86 以外未验证 Provider，不要提交模型、凭据或本机编译产物。
持续推进到最小真实 Prefill+Decode 和可复现 RunReport。
```

## 8. Git 与数据边界

- GitHub 只承载源码、文档、锁文件、测试和紧凑证据。
- `.venv` 约 5 GiB，但不属于项目交付；目标机必须重建。
- checkpoint、HF cache、Codex auth、SSH key、token 和原始大 trace 单独迁移。
- `layer_streaming/providers/*/_build/*.so` 是本机产物并被忽略；目标机重新编译。
- 交接后的真实远端提交以 `git rev-parse HEAD` 为准，不要依赖本文的“交接前基线提交”。
