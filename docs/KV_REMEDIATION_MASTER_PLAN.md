# Cascade-LLM KV 子系统一次性整改总计划

> 本文件是本轮 KV 整改的**唯一总规范和唯一任务入口**。  
> 其他配套文件只是本文件的执行摘录，不增加任何隐藏要求。若内容冲突，以本文件为准。

## 0. 本轮执行结论

本轮目标不是只补几个测试，也不是只写 Quest/KVDrive 接口，而是在**当前工作树、不新建分支、不切换分支、不依赖真实模型权重**的前提下，一次性完成所有当前环境能够完成的 KV 架构整改、逻辑实现和统一验证。

必须完成：

1. 现有 KV 代码全量审计。
2. KV 生命周期、页面所有权、异步 pin、generation/version 契约统一。
3. Fork、COW、Prefix、Beam、Speculative、Rollback 的一致性补强。
4. Quest 风格 page/chunk 索引基础设施和 CPU 参考路径。
5. KVDrive 风格 GPU/CPU/SSD 统一位置索引、模拟分层存储和迁移状态机。
6. 索引、页面、迁移、预取、回收之间的完整闭环。
7. Prefill 与 Decode 执行路径从接口和调度层彻底分离。
8. 建立一次性统一验证工具和机器可读报告。
9. 对受 GPU、模型权重、真实 NVMe 限制的项目明确标记，不伪造通过。
10. 完成最终实现报告、剩余风险和硬件后续命令。

本轮不允许只做计划、只做审计、只建空接口、只生成报告后停止。

---

## 1. 当前系统的准确定位

当前已经具备的基础能力，按现有对话记录包括：

- Paged KV 页面分配和释放。
- `ref_count`、`pin_count` 基础管理。
- Fork/COW/Prefix/Beam/Speculative 的部分生命周期支持。
- `PagedAttentionBackend` 与 `PagedKVKernelBackend` 的初步拆分。
- 部分 SM86 Decode/短 suffix Kernel。
- 1000 token 连续生成和 1000-cycle 所有权测试。
- quiesce 后页面释放、CUDA allocated/reserved 无持续漂移的局部结果。

但当前系统仍应被定义为：

> **Paged KV 页面所有权 Runtime + 部分 Paged Attention Kernel。**

不能定义为完整的可检索、可分层、可跨介质调度的 KV 系统，也不能定义为 Production Qualified。

---

## 2. 问题总清单

### 2.1 生命周期和所有权

1. 页面生命周期虽然已经收口到 KV Runtime，但异常路径、取消路径和并发路径仍未充分证明。
2. `ref_count` 与 `pin_count` 语义必须完全区分：
   - `ref_count` 表示逻辑所有者；
   - `pin_count` 表示暂时不可回收的执行或 IO 使用者。
3. 页面释放依赖 quiesce，说明异步 Kernel/stream/IO 生命周期仍可能发生遗漏。
4. 未明确证明旧 Page ID 在页面复用后不会形成 ABA/悬空句柄问题。
5. 重复 release、重复 unpin、异常退出、部分初始化失败后的状态恢复需要硬断言。
6. 多请求动态 batch 下同时 allocate/fork/COW/release 的竞态尚未形成统一验收。

### 2.2 共享、分叉和回滚

1. Fork 页面共享与首次写入 COW 的边界需要统一。
2. Prefix Cache 的引用和淘汰必须与活跃请求引用分开。
3. Beam 分叉、分支退出、父子序列释放需要确保不影响存活分支。
4. Speculative decode 的 append、commit、partial commit、rollback 必须同时更新 KV 数据、逻辑长度和索引版本。
5. COW 后索引不能继续共享可变状态。
6. 部分页 append、页边界切换和回滚到页中间位置需要专门验证。

### 2.3 Quest 风格索引未完整落地

1. 当前主要仍是分页遍历或全量 KV 扫描。
2. 缺少统一 page/chunk 摘要索引接口。
3. 缺少索引构建、增量更新、选页、版本校验和序列化流程。
4. 缺少全量选择模式作为精确参考路径。
5. 缺少 Top-K/预算式稀疏选择的 Recall、误差和命中率统计。
6. 索引与 Fork/COW/Rollback/Prefix 淘汰尚未闭环。
7. Quest 风格索引不能被误写成“已经完成 Quest 论文全部算法”；本轮先完成可替换的索引基础设施和 CPU reference。

### 2.4 KVDrive 风格分层索引未完整落地

1. 缺少稳定的逻辑 KV Block ID 到物理页面/介质位置映射。
2. 缺少 GPU、CPU、SSD 的统一位置描述和 authoritative copy 规则。
3. 缺少完整迁移状态机、失败恢复和版本提交协议。
4. 缺少索引常驻、数据按需加载的统一读取路径。
5. 缺少预取去重、取消、优先级、容量预留和回收协议。
6. 缺少真实 SSD 时仍可执行的 Mock Tier/Mock IO 验证。
7. 本轮不宣称复现 KVDrive 全部性能优化，只完成其风格所要求的系统骨架和逻辑闭环。

### 2.5 元数据和接口边界

1. 生命周期元数据、索引元数据、数据版本和物理位置尚未统一。
2. Backend 不应直接修改页面所有权和引用计数。
3. Attention 执行、页面管理、索引选择和分层存储必须通过明确接口组合。
4. 页面 ID、逻辑 Block ID、sequence/branch 身份、token range 等概念必须区分。
5. 数据格式、布局、量化信息和索引格式需要显式描述，不能依赖隐式约定。

### 2.6 Prefill/Decode 架构问题

1. 当前 Full Prefill Kernel 明显没有形成适合大 Query 长度的执行映射。
2. 曾出现约 27.1× 差距，不能归因于 Paged KV 的必然开销。
3. Decode 型“少量 Query 扫描长 KV”映射不能直接复用于完整 Prefill。
4. Full Prefill 与 Decode 必须从 Provider、调度和 Kernel 路由层分开。
5. 本轮即使没有足够 GPU，也必须先完成路径拆分和路由测试。
6. CUDA 性能优化和专用硬件调优可因环境暂缓，但不能继续保留错误架构耦合。

### 2.7 调度、淘汰和预取

1. 当前权重预取参数不能代替 KV 独立预取调度器。
2. 被 pin 的页面不得淘汰。
3. 迁移中的页面读取必须有唯一权威版本。
4. 同一页面的并发预取必须去重。
5. 请求取消后 reservation、IO future 和 pin 必须最终回收。
6. 多请求下需要最低限度公平性，不能永久饿死低优先级请求。

### 2.8 验证和可观测性

1. 当前局部 1000-cycle 和单请求测试不足以证明生产可靠性。
2. 缺少统一入口、统一状态、机器可读报告和最小复现序列。
3. 缺少随机状态机测试和故障注入。
4. 缺少逻辑验证、合成 CUDA 验证、真实模型验证之间的明确分层。
5. 缺少 `PASS / FAIL / SKIPPED_WITH_REASON / BLOCKED` 的严格语义。
6. 缺少页面数、引用数、pin 数、迁移数、索引命中率、预取命中率等统一指标。

---

## 3. 本轮范围

### 3.1 必须在本轮完成

以下内容不依赖大模型权重，应一次性完成：

- 代码审计和调用图。
- 核心 KV 契约。
- generation/version 防护。
- 生命周期异常恢复和取消。
- 随机状态机测试。
- Fork/COW/Prefix/Beam/Speculative/rollback 一致性。
- Quest 风格 CPU reference 索引和接口。
- 索引版本与页面版本同步。
- KVDrive 风格统一位置表和模拟分层存储。
- 迁移状态机、Mock IO、预取去重和失败恢复。
- Prefill/Decode Provider 路由拆分。
- 统一验证工具。
- Markdown 和 JSON 报告。

### 3.2 有普通 CUDA 环境时完成

- 合成 Decode 数值对齐。
- 合成 Prefill 数值对齐。
- 多 stream 竞态验证。
- 长循环显存漂移验证。
- 当前 GPU 架构下的基线性能记录。

这些项目不能因为没有模型权重而跳过；只需要随机小 Tensor。

### 3.3 因环境可以暂缓

只有以下内容可以 `SKIPPED_WITH_REASON`：

- 真实 8B/70B 等模型权重数值验证。
- 真实长上下文 benchmark。
- 真实 NVMe Direct IO/GDS 性能。
- 不存在的 GPU 架构专用 Kernel 调优。
- 多卡通信和跨节点 KV。
- 依赖外部数据集的准确率评测。

“功能尚未实现”不能标记为环境跳过。

---

## 4. 硬性执行规则

1. 不新建分支。
2. 不切换分支。
3. 不创建 worktree。
4. 不执行 `git reset --hard`、`git clean`、`git stash`、rebase 或强制 checkout。
5. 不覆盖用户现有未提交修改；必须先读取并融合。
6. 默认不 commit、不 push，只修改当前工作树并生成报告。
7. 开始前保存当前状态和补丁快照，但不得改变工作树。
8. 不允许只做审计后停止。
9. 不允许只添加 TODO、空类或永远返回全量结果的伪实现后宣称完成。
10. Quest 稀疏模式可以先是 baseline scorer，但必须具有真实索引记录、预算选择和统计。
11. Tiered KV 可以使用 Mock SSD，但必须执行真实状态迁移、版本提交和失败回滚逻辑。
12. 所有新增实现必须有测试。
13. 所有失败必须返回非零退出码。
14. 环境受限必须记录具体原因和后续命令，不得标记 PASS。
15. 不通过降低断言、吞异常、删除测试、扩大容差或修改报告获得通过。
16. 不向用户反复询问细节；在不破坏数据的前提下做合理假设并记录。
17. 整个任务尽量在一次 Codex 执行中从审计推进到最终报告。

---

## 5. 开始前工作树保护

Codex 首先执行以下等价操作：

```bash
mkdir -p reports/kv_remediation

git status --short > reports/kv_remediation/preflight_git_status.txt
git diff --binary > reports/kv_remediation/preflight_worktree.patch
git diff --cached --binary > reports/kv_remediation/preflight_index.patch
git ls-files --others --exclude-standard \
  > reports/kv_remediation/preflight_untracked_files.txt

git rev-parse HEAD > reports/kv_remediation/preflight_head.txt
```

规则：

- 这些文件只用于恢复和审计。
- 不执行 stash。
- 若已有修改涉及 KV，Codex 必须基于现状继续，不得覆盖。
- 若仓库不是干净状态，不是中止理由。

---

## 6. 核心架构契约

Codex 必须先审计现有命名，再用现有结构实现以下语义。不要为了名称一致而大范围无关重构。

### 6.1 身份类型

至少区分：

```text
LogicalKVBlockId
    表示某模型、请求/会话、分支、层和 token block 的逻辑身份。

PhysicalPageHandle
    至少包含 physical_page_id + generation。
    任何访问都必须验证 generation。

KVLocation
    表示 tier、device、offset/slot、length、layout、dtype/quant format、version。

IndexRecordId
    表示与逻辑 Block 对应的索引记录，不得仅依赖裸指针。
```

逻辑 ID 与物理页 ID 不能混用。

### 6.2 统一元数据

现有代码可以拆分存储，但语义上至少包含：

```text
logical_block_id
physical_page_handle
sequence/session identity
branch/epoch identity
layer identity
token_start
token_count
capacity
layout
dtype or quant format
data_version
index_version
ref_count
pin_count
ownership_state
locations[]
authoritative_location
inflight_compute
inflight_io
dirty/error flags
```

### 6.3 状态模型

不要使用一个巨型枚举组合所有状态。推荐分为两条正交状态：

#### 所有权状态

```text
FREE
ALLOCATED
RELEASING
ERROR
```

#### 每个介质副本的驻留状态

```text
ABSENT
LOADING
RESIDENT
EVICTING
FAILED
```

迁移必须遵循：

```text
1. 目标位置预留。
2. 源 authoritative version 被 pin 或等价保护。
3. 数据复制。
4. 校验版本和完整性。
5. 原子提交 authoritative_location。
6. 释放源副本，或降为只读 replica。
7. 失败时目标变为 ABSENT/FAILED，源仍保持 authoritative。
```

### 6.4 核心不变量

必须写成运行时断言和测试断言：

```text
ref_count >= 0
pin_count >= 0

FREE 页面必须满足：
- ref_count == 0
- pin_count == 0
- inflight_compute == 0
- inflight_io == 0
- 无有效逻辑映射

旧 generation 的 PhysicalPageHandle 不得访问复用后的页面。

共享页面写入前必须 COW。

page.data_version == index.data_version，
除非索引明确处于 BUILDING 状态且不可被查询。

每个逻辑 Block 在任意时刻必须有且只有一个 authoritative data version。

迁移期间可以存在多个物理副本，
但只有一个版本可被提交为 authoritative。

被 pin 的页面不得进入最终释放或不可恢复淘汰。

取消、异常、OOM、IO 失败和 rollback 最终必须收敛到合法状态。
```

### 6.5 模块职责

#### KV Runtime

唯一负责：

- allocate/release；
- ref/pin；
- generation；
- Fork/COW 所有权；
- 页面状态和最终回收；
- 统一元数据写入入口。

#### Attention Backend

只负责：

- 接收已经解析的页面视图或选择结果；
- 提交计算；
- 正确获取和释放 compute pin；
- 不直接修改 ref_count 或释放页面。

#### Index Backend

只负责：

- build/update/select；
- 索引版本；
- 查询统计；
- Fork 共享和 COW 索引策略；
- 不直接迁移或释放数据页。

#### Tiered Store

只负责：

- 位置表；
- 迁移、加载和淘汰；
- IO pin/reservation；
- authoritative location 提交；
- 不改变逻辑所有权。

#### Scheduler/Coordinator

负责组合：

```text
Query
→ Index select
→ Residency check
→ Prefetch/migrate
→ Compute pin
→ Attention
→ Unpin
→ Optional eviction
```

---

## 7. 总任务分解

## T0：仓库审计与现状地图

### 目标

建立真实代码地图，不依赖对话中的旧结论。

### 操作

1. 搜索所有包含以下语义的文件和符号：
   - allocate/free/release；
   - ref_count/pin_count；
   - page table/block table；
   - fork/COW/prefix/beam/speculative/rollback；
   - paged attention/prefill/decode；
   - Quest/index/select/top-k；
   - tier/offload/SSD/CPU/GPU/prefetch/evict；
   - quiesce/synchronize/event/stream。
2. 画出调用链和写状态的位置。
3. 找出重复管理、绕过 Runtime 的写入和隐式状态。
4. 记录现有测试和缺口。
5. 记录现有用户修改，不覆盖。

### 产物

```text
docs/KV_CURRENT_STATE_AUDIT.md
reports/kv_remediation/kv_symbol_inventory.txt
reports/kv_remediation/kv_test_inventory.txt
```

### 验收

- 每个核心状态字段都有唯一或明确的写入者列表。
- 每个页面释放入口都有调用方。
- 每个 Backend 的职责边界有记录。
- 不允许只列文件名，不分析调用关系。

---

## T1：建立基线和统一报告状态

### 目标

在修改前知道什么原本通过、什么原本失败。

### 操作

1. 找到现有最小 CPU 测试、CUDA 合成测试和项目测试入口。
2. 执行不需要模型权重的现有测试。
3. 将结果记录为 baseline，不因现有失败立即中止整改。
4. 建立状态枚举：

```text
PASS
FAIL
SKIPPED_WITH_REASON
BLOCKED
```

### 规则

- `SKIPPED_WITH_REASON` 只用于缺 GPU、权重、NVMe 或外部依赖。
- 未实现功能必须是 FAIL，不得 SKIP。
- BLOCKED 必须写清楚被哪个失败阻塞。

### 产物

```text
reports/kv_remediation/baseline.md
reports/kv_remediation/baseline.json
```

---

## T2：冻结 KV 契约

### 目标

先统一身份、版本、状态和职责，再修改实现。

### 操作

1. 对照第 6 节映射到现有类型。
2. 尽量复用现有命名，必要时增加 adapter。
3. 增加 `generation` 或等价防 ABA 机制。
4. 增加 `data_version` 和 `index_version`。
5. 明确 logical block 与 physical page 的映射。
6. 明确 ref/pin/inflight 的含义。
7. 明确释放、迁移和错误恢复条件。

### 产物

```text
docs/KV_ARCHITECTURE_CONTRACT.md
```

### 验收

- 契约有对应代码类型或明确迁移计划。
- 不存在多个模块直接修改同一所有权字段而无统一入口。
- 后续任务以该契约为准。

---

## T3：生命周期与故障恢复

### 目标

使页面在成功、取消和失败路径下都能最终回收。

### 操作

1. 收口 allocate/release/ref/pin。
2. 增加旧 generation 句柄拒绝逻辑。
3. 明确 double release、double unpin 的错误行为。
4. 模拟以下故障点：
   - allocate 中途失败；
   - COW 分配失败；
   - Kernel 提交失败；
   - Kernel 完成回调失败；
   - 请求取消；
   - OOM；
   - IO submit 失败；
   - IO completion 失败。
5. 每个故障点必须验证状态收敛。
6. 增加 quiesce 超时和诊断信息，不能无限等待无信息。

### 验收

- 10 万级随机生命周期操作无泄漏、负计数或悬空访问。
- 相同随机种子可复现。
- 失败报告包含最小操作序列。

---

## T4：Fork/COW/Prefix/Beam/Speculative 一致性

### 目标

使共享、分叉和回滚形成同一所有权协议。

### 操作

1. Fork：新分支共享只读页面并增加正确引用。
2. COW：首次写入只复制必要页，未修改页保持共享。
3. Prefix：缓存持有者和活跃请求持有者分别计数或可区分追踪。
4. Beam：父分支退出不释放子分支仍使用页面。
5. Speculative：
   - append；
   - full commit；
   - partial commit；
   - rollback；
   - 跨页 rollback。
6. 数据长度、页面表、data_version 和 index_version 同步变化。

### 验收

- 任一分支修改不污染其他分支。
- 任一分支退出不破坏存活分支。
- rollback 后全量 reference 结果与未 speculative 路径一致。
- 页面和索引都无泄漏。

---

## T5：Quest 风格索引基础设施

### 目标

完成可替换、可验证的 page/chunk 索引，不把索引仅停留在概念层。

### 必须提供的接口语义

```text
build(block_data, metadata) -> index_record
update_append(index_record, appended_data, new_version)
select(query, candidates, budget, mode) -> selection_result
fork_ref(index_record)
cow_clone(index_record)
rollback(index_record, target_token_count, target_version)
serialize/deserialize(index_record)
validate(index_record, page_metadata)
```

### 模式

1. `full`：选择全部合法 KV Block，保持原始顺序。
2. `topk` 或 `budget`：根据摘要索引选择有限 Block。
3. `debug_exact`：可选，用 reference score 检查 selector。

### 最低实现要求

- CPU reference。
- 真实索引记录，不得每次查询重新遍历原始全部 K/V 后假装索引。
- 至少一种稳定摘要和 scorer。
- 记录候选数、选择数、索引耗时和 Recall 相关统计。
- 全量模式必须与非索引全量 Attention 等价。
- 稀疏模式不设伪造准确率门槛；输出可重复的误差和 Recall 数据。

### 一致性

- append 更新索引版本。
- COW 后可变索引隔离。
- Fork 可以共享不可变索引记录。
- rollback 同步恢复 token range 和版本。
- 页面被淘汰不等于索引必须淘汰；策略要显式。

### 验收

- `index_version == data_version` 才能参与查询。
- 版本不一致时必须重建、更新或明确失败。
- 序列化往返一致。
- full 模式精确对齐。

---

## T6：KVDrive 风格统一分层位置索引

### 目标

实现不依赖真实 SSD 的 GPU/CPU/SSD 统一逻辑层。

### 操作

1. 定义 `KVLocation`。
2. 定义 logical block → locations[] + authoritative_location。
3. 提供 Mock GPU、Mock CPU、Mock SSD backend。
4. 使用真实内存缓冲区或临时文件模拟数据移动，不能只改状态不复制数据。
5. 实现：
   - put；
   - get；
   - migrate；
   - prefetch；
   - evict；
   - cancel；
   - failure rollback；
   - checksum/version validation。
6. 支持一个 authoritative copy 和可选只读 replica。
7. 数据迁移提交必须原子化到元数据层。

### 验收

- GPU→CPU→SSD→CPU→GPU 往返数据一致。
- 任一迁移阶段失败，源数据仍可读。
- 迁移期间并发读取返回合法版本或明确等待，不返回半写数据。
- 同一 Block 并发预取去重。

---

## T7：淘汰、预取和调度闭环

### 目标

将 Index、Tiered Store 和 Attention 串起来。

### 操作

1. 根据 selector 输出检查 residency。
2. 对缺失页面提交异步预取。
3. 预取使用 reservation 和 IO pin。
4. 对重复请求去重。
5. 请求取消传播到尚未开始的 IO；已开始 IO 完成后回收。
6. 页面 compute pin 期间不可淘汰。
7. 实现最小可用淘汰策略：
   - LRU/clock 或现有策略；
   - 明确排除 pinned/inflight 页面；
   - 记录淘汰原因。
8. 实现最低公平性：等待队列不能永久饿死。

### 验收

- selection → prefetch → attention view → unpin 路径可用 Mock backend 端到端运行。
- 取消后无残留 reservation、pin 和 future。
- 容量不足时返回明确错误或执行合法淘汰，不破坏状态。

---

## T8：Prefill/Decode 路径拆分

### 目标

消除 Full Prefill 继续走 Decode 型执行映射的架构问题。

### 操作

1. 在 Provider/Dispatcher 层显式区分：
   - Full Prefill；
   - Chunked Prefill；
   - Decode；
   - Short Suffix。
2. 建立独立 capability 查询和 fallback。
3. Full Prefill 禁止静默路由到仅适合 Q=1～4 的 Decode Kernel。
4. 无专用 Full Prefill Kernel 时，路由到正确但可能较慢的 reference/通用实现，并明确记录。
5. Quest 稀疏选择默认只用于适合的 Decode/长上下文路径，不替代 Full Prefill。
6. 添加路由单元测试，不要求模型权重。

### 验收

- 不同 workload shape 有明确路由结果。
- Full Prefill 不再调用 Decode-only Kernel。
- fallback 有状态和原因。
- 当前 27.1× 问题被归类为待硬件优化，而不是继续隐藏在通用路径中。

---

## T9：统一一次性验证工具

### 目标

建立一个命令完成所有当前可完成验证。

### 必须新增或完善

```bash
python tools/validate_kv_stack.py --profile logic
python tools/validate_kv_stack.py --profile cuda-synthetic
python tools/validate_kv_stack.py --profile full
```

若项目已有等价工具，应扩展而不是重复创建。

### profile 语义

#### logic

- 不需要模型权重。
- 不强制需要 CUDA。
- 覆盖生命周期、共享、索引、分层迁移、故障注入和调度。
- 本轮必须全部通过。

#### cuda-synthetic

- 只需要 CUDA 和随机 Tensor。
- 覆盖合成 Prefill/Decode、stream 和显存漂移。
- 无 CUDA 时可 `SKIPPED_WITH_REASON`。

#### full

- 包含真实权重、真实长上下文和真实 NVMe。
- 缺环境时逐项跳过，不影响 logic 结论。

### 输出

```text
reports/kv_validation.json
reports/kv_validation.md
reports/kv_validation_failures/<case-id>/seed.txt
reports/kv_validation_failures/<case-id>/repro.json
```

### JSON 至少包含

```text
case_id
name
profile
status
reason
duration
seed
environment
metrics
artifacts
```

---

## T10：最终审查和报告

### 目标

由独立只读审查 Agent 或主 Agent 的独立审查阶段检查实现，不再修改结论掩盖问题。

### 审查内容

1. 所有权字段是否仍有绕过 Runtime 的写入。
2. generation 是否在所有访问路径检查。
3. 数据版本和索引版本是否可能静默失配。
4. 迁移失败是否可能丢失 authoritative copy。
5. pin 是否可能永久不归还。
6. Full Prefill 是否仍可能路由到 Decode-only Kernel。
7. 测试是否真实触发故障，而不是只测 happy path。
8. SKIP 是否被滥用。

### 最终产物

```text
docs/KV_REMEDIATION_IMPLEMENTATION_REPORT.md
reports/kv_remediation/final_git_diff_stat.txt
reports/kv_remediation/final_review.md
reports/kv_validation.md
reports/kv_validation.json
```

实现报告必须分为：

- 已实现并验证；
- 已实现但仅模拟验证；
- 仅完成接口；
- 因环境暂缓；
- 仍失败；
- 后续硬件命令。

---

## 8. 完整验证矩阵

### 8.1 基础与生命周期

| ID | 验证项 | 环境 | 通过标准 |
|---|---|---|---|
| V00 | 预修改状态快照 | CPU | status、diff、HEAD 均保存 |
| V01 | 现有无权重测试基线 | CPU | 结果被记录，不静默忽略失败 |
| V02 | allocate/release 闭环 | CPU | 最终页面全部回池 |
| V03 | ref_count 非负 | CPU | 任意操作序列不为负 |
| V04 | pin_count 非负 | CPU | 任意操作序列不为负 |
| V05 | double release/unpin | CPU | 明确错误且状态不损坏 |
| V06 | generation/ABA | CPU | 旧句柄访问被拒绝 |
| V07 | quiesce 收敛 | CPU/Mock | inflight、pin 最终清零 |
| V08 | 请求取消 | CPU/Mock | 资源最终释放 |
| V09 | 模拟 OOM | CPU/Mock | 状态原子恢复，后续可继续 |
| V10 | 模拟 Kernel 失败 | CPU/Mock | compute pin 被归还 |
| V11 | 模拟 IO submit 失败 | CPU/Mock | 不错误提交位置 |
| V12 | 模拟 IO completion 失败 | CPU/Mock | 源 authoritative 仍可读 |
| V13 | 并发 allocate/fork/free | CPU | 无死锁、负计数和损坏 |
| V14 | 10 万随机状态操作 | CPU | 零泄漏、零悬空、可复现 |
| V15 | 最小复现缩减 | CPU | 失败时生成 seed 和 repro |

### 8.2 共享和回滚

| ID | 验证项 | 环境 | 通过标准 |
|---|---|---|---|
| V16 | Fork 共享 | CPU | 初始共享正确增加引用 |
| V17 | COW 隔离 | CPU | 仅写页复制，其他分支不受影响 |
| V18 | Prefix 引用 | CPU | 缓存和请求释放顺序均正确 |
| V19 | Prefix 淘汰 | CPU | 活跃引用存在时不能释放数据 |
| V20 | Beam 分叉 | CPU | 父子和兄弟分支隔离 |
| V21 | Beam 分支退出 | CPU | 存活分支数据不丢失 |
| V22 | Speculative commit | CPU | 数据、长度、版本一致 |
| V23 | Speculative partial commit | CPU | 保留前缀正确 |
| V24 | Speculative rollback | CPU | 恢复到目标版本 |
| V25 | 跨页 rollback | CPU | 页表和索引都正确 |
| V26 | 页中 append/rollback | CPU | token range 不错位 |

### 8.3 Quest 风格索引

| ID | 验证项 | 环境 | 通过标准 |
|---|---|---|---|
| V27 | 索引 build | CPU | 生成稳定记录和版本 |
| V28 | full select | CPU | 选择全部合法页且顺序正确 |
| V29 | full 模式数值 | CPU | 与无索引 reference 一致 |
| V30 | budget/top-k select | CPU | 严格遵守预算并稳定输出 |
| V31 | 稀疏统计 | CPU | 输出候选、选择、Recall/误差统计 |
| V32 | append 增量更新 | CPU | index_version 跟随 data_version |
| V33 | Fork 索引共享 | CPU | 不可变记录安全共享 |
| V34 | COW 索引隔离 | CPU | 修改不污染其他分支 |
| V35 | rollback 索引恢复 | CPU | token range 和版本一致 |
| V36 | 版本失配拒绝 | CPU | 不使用过期索引 |
| V37 | 序列化往返 | CPU | 内容和版本一致 |
| V38 | 空序列/单页/尾页 | CPU | 边界行为明确 |

### 8.4 KVDrive 风格分层存储

| ID | 验证项 | 环境 | 通过标准 |
|---|---|---|---|
| V39 | logical→location 映射 | CPU/Mock | 映射稳定且可校验 |
| V40 | GPU→CPU 模拟迁移 | CPU/Mock | 数据一致 |
| V41 | CPU→SSD 模拟迁移 | CPU/Mock | 临时文件或真实 buffer 数据一致 |
| V42 | SSD→CPU→GPU 往返 | CPU/Mock | checksum/version 一致 |
| V43 | authoritative 唯一性 | CPU/Mock | 任意时刻唯一提交版本 |
| V44 | 迁移中并发读取 | CPU/Mock | 合法等待或读取稳定版本 |
| V45 | 迁移失败回滚 | CPU/Mock | 源数据仍可读 |
| V46 | pinned 页面淘汰冲突 | CPU/Mock | 淘汰被拒绝或延迟 |
| V47 | 预取去重 | CPU/Mock | 同页只产生一个实际 IO |
| V48 | 预取取消 | CPU/Mock | reservation/pin 最终释放 |
| V49 | 容量不足 | CPU/Mock | 合法淘汰或明确失败 |
| V50 | 淘汰后重载 | CPU/Mock | 数据和索引仍一致 |
| V51 | 元数据原子提交 | CPU/Mock | 不暴露半迁移状态 |

### 8.5 调度和 Provider

| ID | 验证项 | 环境 | 通过标准 |
|---|---|---|---|
| V52 | selection→prefetch→view | CPU/Mock | 端到端闭环 |
| V53 | compute pin 生命周期 | CPU/Mock | 调用前后严格配对 |
| V54 | 多请求预取公平性 | CPU/Mock | 无永久饿死 |
| V55 | 请求取消传播 | CPU/Mock | 后续阶段不继续消费已取消请求 |
| V56 | Full Prefill 路由 | CPU | 不进入 Decode-only Kernel |
| V57 | Decode 路由 | CPU | 进入正确 Kernel/fallback |
| V58 | Chunked Prefill 路由 | CPU | 路由和 capability 明确 |
| V59 | 无专用 Kernel fallback | CPU | 正确路径并记录原因 |

### 8.6 CUDA 合成验证

| ID | 验证项 | 环境 | 通过标准 |
|---|---|---|---|
| V60 | 合成 Decode 数值 | CUDA | 与 reference 误差达标 |
| V61 | 合成 Full Prefill 数值 | CUDA | 与 reference 误差达标 |
| V62 | 合成 Chunked Prefill | CUDA | 与 reference 误差达标 |
| V63 | 多 stream 竞态 | CUDA | 重复运行稳定 |
| V64 | CUDA allocated 漂移 | CUDA | 长循环无持续增长 |
| V65 | CUDA reserved 漂移 | CUDA | 解释缓存行为且无非预期增长 |
| V66 | Kernel 失败后恢复 | CUDA | 页面/pin 可继续使用或回收 |
| V67 | 当前硬件性能基线 | CUDA | 记录数据，不伪造跨硬件结论 |

### 8.7 真实环境后续验证

| ID | 验证项 | 环境 | 本轮状态规则 |
|---|---|---|---|
| V68 | 真实模型 logit/Top-1 | 模型权重 | 缺权重则 SKIPPED_WITH_REASON |
| V69 | 真实 8B 长生成 | 模型权重+GPU | 缺环境则 SKIPPED_WITH_REASON |
| V70 | 真实长上下文 Prefill | 权重+GPU | 缺环境则 SKIPPED_WITH_REASON |
| V71 | 真实 Quest 准确率 | 权重+数据集 | 缺环境则 SKIPPED_WITH_REASON |
| V72 | 真实 NVMe 吞吐/延迟 | NVMe | 缺设备则 SKIPPED_WITH_REASON |
| V73 | IO/计算流水重叠率 | GPU+NVMe | 缺环境则 SKIPPED_WITH_REASON |
| V74 | SM86 专用调优 | SM86 | 非对应硬件则跳过 |
| V75 | 其他架构专用调优 | 对应 GPU | 非对应硬件则跳过 |
| V76 | 大规模动态 batch | 权重+服务环境 | 缺环境则跳过 |

### Logic Gate

`--profile logic` 至少必须覆盖 V02～V59 中所有不要求 CUDA 的项目，并全部 PASS。  
任何功能未实现都必须 FAIL，不能 SKIP。

### Production Gate

只有以下同时满足才能标记 Production Qualified：

- logic 全部 PASS；
- cuda-synthetic 全部适用项 PASS；
- 真实模型数值 PASS；
- 真实长上下文性能达到项目明确目标；
- 真实 SSD 路径稳定；
- 多请求压力和取消测试 PASS。

本轮预计结论最多是 KV Stack Beta 或相应中间状态，不能提前标记 Production Qualified。

---

## 9. Agent 协作方案：不建分支情况下的最高效方式

同一工作树不能让多个写 Agent 无约束并发修改公共文件。因此采用：

### 9.1 主控 Agent

负责：

- 保存工作树快照；
- 调度只读审计；
- 冻结契约；
- 按顺序安排写任务；
- 运行统一验证；
- 生成最终报告。

### 9.2 第一阶段：只读子 Agent 并行

最多四个，只读，不修改文件：

1. Runtime Audit Agent
   - 生命周期、ref/pin、generation、异常回收。
2. Sharing/Index Audit Agent
   - Fork/COW/Prefix/Beam/Speculative、Quest 相关代码。
3. Tiering/IO Audit Agent
   - CPU/SSD/offload/prefetch/eviction。
4. Kernel/Validation Audit Agent
   - Prefill/Decode 路由、Kernel 和测试入口。

每个 Agent 返回：

```text
相关文件和符号
当前实现
缺失项
风险
建议最小修改范围
测试入口
```

### 9.3 第二阶段：单写者契约冻结

只允许主控 Agent 修改公共契约和公共类型。

### 9.4 第三阶段：写任务按依赖推进

在同一工作树下，默认串行写入：

```text
Runtime
→ Sharing/COW
→ Index
→ Tiered Store
→ Scheduler
→ Provider routing
→ Validation harness
```

只有确认目录完全不重叠时，才允许两个子 Agent 并行；公共类型、注册表、构建文件和统一测试入口必须由主控 Agent 独占修改。

### 9.5 最终独立审查

使用只读 Review Agent 检查：

- 竞态；
- 引用泄漏；
- 过期索引；
- 迁移丢数据；
- 错误 SKIP；
- Prefill 错误路由。

Review Agent 先报告，主控 Agent 再修复；修复后重新完整执行验证。

---

## 10. 一次性执行顺序

Codex 不应把任务拆成需要用户多次重新提示的阶段。单次执行中按以下顺序持续推进：

```text
Step 1  保存工作树快照
Step 2  并行只读审计
Step 3  写入审计文档
Step 4  建立 baseline
Step 5  冻结契约和公共类型
Step 6  修复 Runtime 生命周期
Step 7  修复 Fork/COW/Prefix/Beam/Speculative
Step 8  实现 Quest 风格 CPU 索引
Step 9  实现 KVDrive 风格 Mock Tiered Store
Step 10 串联预取、淘汰和调度
Step 11 拆分 Prefill/Decode 路由
Step 12 建立统一验证工具
Step 13 执行 logic 全量验证并修复
Step 14 有 CUDA 则执行 cuda-synthetic 并修复
Step 15 执行只读独立审查
Step 16 再次运行完整验证
Step 17 生成最终实现报告
```

不得在 Step 2、Step 5 或 Step 12 后因“任务较大”自行停止。

---

## 11. Codex 修改范围原则

Codex 应先根据仓库结构确定真实目录，不要假设下列路径一定存在。修改遵循：

1. 优先修改现有 KV Runtime、backend、kernel 和 tests。
2. 避免为计划强行创建大量新抽象层。
3. 公共契约保持最小但完整。
4. 不重构无关权重加载、Web UI、模型定义和 CLI。
5. 不改变用户可见参数语义，除非旧语义本身错误；此时必须写兼容层和迁移说明。
6. 新功能默认可通过 feature flag 或明确 backend 选择启用。
7. reference 路径必须始终可用于验证优化路径。
8. 所有错误必须包含可定位上下文：page/block、generation、version、request/branch、tier。

---

## 12. Codex 完成时必须给出的结果

最终回复和文档必须直接列出：

1. 修改了哪些文件。
2. 每个任务 T0～T10 的状态。
3. 每个 Vxx 验证项的状态。
4. logic profile 是否全部通过。
5. cuda-synthetic 是否执行；未执行的具体原因。
6. 哪些项因模型权重/NVMe/GPU 暂缓。
7. 当前仍存在的失败和风险。
8. 用户下一步应执行的命令。
9. 不得只说“测试通过”而不提供报告路径。

---

## 13. 用户操作步骤

### 第一步：把本计划包复制进仓库

目标目录：

```text
/disk2/home/guest/lianghan/repos/Cascade-LLM/docs
```

至少应存在：

```text
docs/KV_REMEDIATION_MASTER_PLAN.md
docs/KV_CODEX_EXECUTION_PROMPT.md
docs/KV_VALIDATION_MATRIX.md
docs/KV_ARCHITECTURE_CONTRACT_TEMPLATE.md
```

其中主计划是唯一总规范。

### 第二步：进入仓库

```bash
cd /disk2/home/guest/lianghan/repos/Cascade-LLM
```

### 第三步：不要新建分支

只检查状态：

```bash
git status --short
```

不要提前执行 reset、stash、clean 或 checkout。

### 第四步：启动 Codex

交互模式：

```bash
codex
```

然后完整粘贴：

```text
docs/KV_CODEX_EXECUTION_PROMPT.md
```

更适合一次性执行的非交互方式：

```bash
codex exec "$(cat docs/KV_CODEX_EXECUTION_PROMPT.md)"
```

Codex 必须在仓库根目录运行，并具有读取、编辑仓库和执行测试所需权限。

### 第五步：Codex 完成后只做检查

```bash
git status --short
git diff --stat
sed -n '1,240p' docs/KV_REMEDIATION_IMPLEMENTATION_REPORT.md
sed -n '1,260p' reports/kv_validation.md
```

然后执行：

```bash
python tools/validate_kv_stack.py --profile logic
```

有 CUDA 时再执行：

```bash
python tools/validate_kv_stack.py --profile cuda-synthetic
```

### 第六步：不要根据一句总结判断是否完成

只有满足以下条件才接受本轮结果：

- T0～T10 有逐项状态；
- V02～V59 的适用逻辑项全部 PASS；
- 缺硬件项目是 `SKIPPED_WITH_REASON`；
- 没有未解释的 BLOCKED；
- 没有把空接口当作实现；
- 最终报告明确当前里程碑，不冒充 Production Qualified。

---

## 14. 里程碑定义

### KV Runtime V1 RC

- 页面所有权、ref/pin、Fork/COW 基础可用。
- 不代表索引和分层存储完成。

### Indexed KV Alpha

- Quest 风格索引基础设施完成。
- full 模式精确验证通过。
- 稀疏模式具有可测指标。

### Tiered KV Alpha

- GPU/CPU/SSD 统一逻辑位置和 Mock 迁移闭环完成。
- 不代表真实 NVMe 性能通过。

### KV Stack Beta

- logic profile 全部通过。
- Prefill/Decode 路由拆分完成。
- 合成 CUDA 适用项通过或有明确硬件限制。

### Production Qualified

- 真实权重、长上下文、多请求、真实 SSD 和目标 GPU 性能均通过。

---

## 15. 本轮完成判定

本轮 Codex 任务只有在以下条件同时满足时才算完成：

1. 审计不是最终产物，而是已经推动代码整改。
2. 核心契约已经映射到真实代码。
3. logic profile 可一条命令执行。
4. 所有无硬件依赖的验证项都已实现并通过。
5. Quest 风格索引不再只是设计文档。
6. KVDrive 风格分层不再只是位置枚举，而有真实 Mock 数据移动和失败恢复。
7. Full Prefill 与 Decode 路由不再混用。
8. 最终报告完整且没有隐藏后续任务。
9. 所有暂缓项都明确说明所需环境和执行命令。
10. 用户无需再次询问“还有哪些没有做”。
