# Cascade-LLM 开发文档（本地）

> 2026-08-01 更新：M1～M5.1 的核心接口、正确性、稳定流水线、统一 benchmark 和 SM86 CUTLASS W8A16 provider 已落地；M6 已完成静态整层 Transformer 常驻、独立 LM Head placement、真实 8B INT8 checkpoint 转换和第一轮显存/延迟曲线。真实 W8A16 全常驻可达到约 46 ms/token，但 provider 与 fallback 的严格逐层数值门槛及完整 M6A 长矩阵仍未关闭。详见 `docs/M6_VALIDATION.md`。

> 本文档只保存在本地，不上传 GitHub。它描述当前源码状态，重点记录设计边界、已知问题和下一阶段实现顺序。

## 1. 项目定位

Cascade-LLM 是单机、单 GPU、单请求优先的 CPU-resident LLM 推理原型。完整权重常驻 CPU，GPU 只保留当前计算需要的权重、常驻参数和 KV Cache。运行时使用独立 CUDA Copy Stream 与 Compute Stream，使 H2D 传输和当前矩阵计算尽可能重叠。

当前不是通用推理服务器，也不是完整的 W8A8 高性能 kernel 实现。它首先验证的是：在 GPU 显存不足以容纳模型时，能否用较小的 GPU 权重缓冲区完成正确推理，并通过异步流水线降低等待。

## 2. 当前支持范围

### 已实现的模型路径

1. **BF16 Llama 路径**
   - `LlamaModelAdapter` 从 `config.json` 生成 `ModelGeometry` 和执行计划。
   - 不再包含固定 8B/70B geometry builder；所有尺寸来自 config。
   - 支持 `matrix`、`matrix_group`、`layer` 三种传输粒度。
   - 支持完整词表常驻或 Embedding/LM Head 词表流式模式。

2. **统一 dense/量化路径**
   - BF16 和 FP16 dense backend。
   - INT8 symmetric per-channel/per-group，group size 32/64/128，BF16/FP16 scale 与 activation。
   - packed INT4 symmetric per-group；低四位保存第一个值，`0..15` 映射到 `-8..7`。
   - Transformer、Embedding、LM Head、Norm 可以分别选择 dtype。
   - INT8/INT4 在 GPU 可复用 workspace 解量化后调用标准 `F.linear`，报告中始终带 `_fallback`。

3. **Checkpoint 与合成验证**
   - `CheckpointManifest` 支持单文件、index + 多 shard、alias/tied weight。
   - 在 CPU/GPU 大内存分配前校验 key、shape、dtype、shard、data offset 和量化 metadata。
   - `tools/generate_tiny_checkpoint.py` 生成 BF16/FP16/INT8/INT4、单文件/多 shard 和未量化 reference。
   - `MemoryPlanner.plan_metadata_only()` 可模拟 8B/70B/100B 以上布局而不创建权重。

### 已实现的存储模式

- `full_pinned`：完整 CPU arena 使用锁页内存，可直接异步 H2D；代价是需要锁定接近完整权重的 RAM。
- `pinned_staging`：完整权重保存在普通 pageable CPU arena，仅使用少量锁页 staging slot；后台线程先做 pageable→pinned copy，再提交 H2D。

### 已实现的推理功能

- 多 token prefill 和后续单 token decode。
- Llama RoPE、GQA、RMSNorm、SwiGLU、SDPA。
- Embedding 按 token 行提取。
- LM Head 按词表行分块，逐块计算 logits，并在线合并全局 Top-k。
- 聊天模板、EOS/EOT 停止、温度/top-k/top-p/min-p、重复惩罚、presence/frequency penalty、多轮会话保存。

## 3. 代码结构与调用链

```text
tools/run_llama31.py
tools/chat_llama31_70b_int8.py
        │
        ├─ AutoConfig / AutoTokenizer（只读本地 checkpoint）
        ├─ ModelAdapter.build_execution_plan()
        ├─ CheckpointManifest.validate()
        ├─ MultiDtypeWeightStore
        ├─ store.load_checkpoint()
        ├─ ResidentDeviceArena（常驻 norm/词表等）
        ├─ MixedDtypeRuntime
        └─ Llama31DecodeExecutor.begin → runtime.run → finish

layer_streaming/
  adapter.py    config/geometry、用户执行策略、checkpoint 元数据校验
  specs.py      WeightSpec、QuantizationSpec 和 dtype/对齐规则
  checkpoint.py 单文件/多 shard manifest、alias 和 header/data 校验
  execution_plan.py 多 dtype byte offset、TransferTensor/TransferUnit
  backends.py   dense、INT8/INT4 fallback、fused provider 注册与能力检查
  multi_dtype_store.py 多 region CPU arena、full-pinned/staging
  mixed_runtime.py 通用 byte slot、backend dispatch、双 stream
  mixed_vocab.py 混合 dtype Embedding/LM Head streaming
  plan.py       旧 BF16 兼容计划
  weight_store.py  BF16 CPU arena、full_pinned/staging
  int8.py       INT8 计划、INT8 store、解量化双缓冲
  runtime.py    BF16 双流双缓冲
  pipeline.py   常驻 producer/H2D scheduler/compute consumer 和有界队列
  kv_cache.py   预分配 block arena、请求 handle、回收和越界检查
  memory.py     CPU/pinned/GPU 启动内存预算与容量预检
  reporting.py  JSON RunReport schema、profile 聚合和硬件元数据
  stability.py  CUDA/pinned/thread/event/KV/queue 快照与趋势判定
  benchmarking.py 版本化 benchmark case/sample/suite schema
  llama31.py    Transformer 层执行与 KV cache
  vocab.py      词表流式 Embedding/LM Head
  chat.py       采样和会话辅助逻辑
```

一次 decode 的基本时序：

1. `begin()` 从 Embedding 得到 hidden states，并根据 KV cache 计算 position ids。
2. runtime 为前几个 transfer unit 创建 CPU source Future。
3. Copy Stream 将 unit 写入 GPU slot，记录 `ready_event`。
4. Compute Stream 等待 ready event，调用 executor 计算当前 unit。
5. 计算完成后记录 `free_event`；相同 slot 被下一轮 H2D 复用前等待该 event。
6. 所有 Transformer unit 完成后，`finish()` 执行 final RMSNorm 和 LM Head Top-k。

## 4. 显存与缓冲区模型

通用路径使用 byte-addressed device slot，slot 内按 256-byte alignment 放置权重、scale 和 zero-point；每个 slot 配一个按最大 linear matrix 预分配的解量化 workspace：

```text
GPU slot = raw transfer bytes + BF16/FP16 dequant workspace
```

workspace 在一次 unit 内按访问顺序复用。权重视图不能从 callback 中逃逸，因为对应 slot 会在后续 unit 重新写入。

词表流式模式会复用 Transformer 的 BF16 workspace；LM Head 每块约 128 MiB，计算结束后只留下当前全局 Top-k，不保存完整 logits。`return_full_logits=True` 会额外申请 `[batch, seq, vocab]` 的 FP32 tensor，应仅用于小规模验证。

## 5. 目前存在的漏洞和风险

以下问题是当前源码的主要限制，按优先级分为 P0（影响正确性/可用性）、P1（影响稳定性或性能）、P2（工程完善）。

### P0：正确性与可用性

#### 5.1 KV Cache 的 O(T²) 追加（已修复）

当前 `KVCacheManager` 在请求开始时按最大上下文分配连续 block run，append 只写预分配 view，不再使用 `torch.cat`。已实现最大长度、固定地址、reset、release、block 复用和可控容量错误。第一版仍是单请求优先，未实现非连续 block table kernel 和 continuous batching。

**修复方向：**实现 paged/block KV cache；按 layer、head、block 管理 K/V，预先限制最大上下文，并在 batch 请求结束时归还 block。

#### 5.2 Batch 支持不完整

Executor 只声明支持“无 padding 且序列长度相同”的 batch。没有 attention mask、每请求 position、不同长度 prefill、独立 EOS、独立 sampling 状态和请求级 KV cache。改变 batch size 或混合 prefill/decode 可能产生错误结果。

**修复方向：**引入 request table、cu-seqlens/attention mask、每请求 cache handle，并将 scheduler 与模型执行解耦。

#### 5.3 INT8/INT4 性能 kernel

当前实现把 INT8 或 packed INT4 解量化为 BF16/FP16 weight，再执行 A16 `F.linear`。解量化和 A16 GEMM 都计入计算路径，不能代表 Tensor Core W8A8/W4A16 性能。

**当前边界：**INT8/INT4 fallback 仍按上述方式执行。SM86 的
`fused_w8a16` 已有真实 CUTLASS provider：symmetric per-channel 支持
M>=1，symmetric per-group 支持 group size 32/64/128 的 M=1 decode；
BF16/FP16 activation 和 scale 均可用，且不申请完整 A16 weight
workspace。per-group prefill、W4A16 和 W8A8 仍未完成，选择不支持的
phase/格式会在启动资格检查阶段失败。详见 `docs/CUTLASS_PROVIDER.md`。

#### 5.4 Checkpoint 几何和命名大量硬编码（Llama 已修复）

Llama 的 hidden、layer、KV head、head dim、vocab 和上下文限制已经由 `ModelAdapter` 从 config 生成；safetensors 在 arena/GPU 分配前校验全部 key、shape 和 dtype，并完整报告缺失、冗余和冲突。Qwen、Mistral 仍不在当前阶段范围内。

**修复方向：**由 `config.json` 生成 geometry，由 safetensors index 校验 key、shape、dtype，再选择模型适配器。

### P1：性能和稳定性

#### 5.5 当前流水线没有真正的水位调度（固定窗口已修复）

当前 runtime 已拆为常驻 SourceProducer、H2DScheduler 和主线程 ComputeConsumer，使用 `source_queue`、`ready_queue`、`free_slot_queue` 三条有界队列；slot 支持 1..4，prefetch depth 支持 1..8，并记录队列容量、水位和等待。当前阶段保留用户显式策略，不做在线自动 granularity/placement 改写。

**修复方向：**增加 scheduler queue、ready/free byte accounting、prefetch window、超时和 fallback；根据最近窗口的 H2D/compute EWMA 动态改变预取深度与 unit 分组。

#### 5.6 host 端仍有串行等待点（已修复第一版）

Future 等待已经移入常驻 SourceProducer；H2D scheduler 独立消费 CPU source，主线程只按序消费 ready unit。source prepare wait、ready wait、free slot wait 和 queue depth 已进入 profile/RunReport。

**修复方向：**采用生产者/消费者队列，限制内存深度，记录 source wait；对 full-pinned 直接路径和 staging 路径分别调度。

#### 5.7 Embedding 路径强制同步 Copy Stream（已修复）

Embedding 现在把一次输入的全部 token row gather 到启动时预分配的连续 pinned buffer，执行单次 H2D，并通过固定 `embedding_ready_event` 让 compute stream 等待；不再按小 batch 同步 Copy Stream。CLI 在 MemoryPlanner 中按最大 prefill/context 同步预算该 pinned buffer。

**修复方向：**为每个 embedding staging block 使用 event，或预先将整个 input 的行 gather 到一个 pinned buffer 后一次 H2D。

#### 5.8 事件和临时对象每 token 大量创建（核心路径已修复）

Transformer runtime 与流式 LM Head 的 ready/free/timing events 已在初始化时创建并复用。INT8 profile 模式中的逐矩阵 dequant timing event 仍是仅在显式 profile 时创建，后续统一进 profiling resource pool。

**修复方向：**建立 event pool、固定环形 event slots，并区分 profile/non-profile 两种低开销路径。

#### 5.9 显存和锁页内存没有启动前预检（已修复第一版）

`MemoryPlanner.preflight()` 已预算 checkpoint arena、pinned staging、transfer slots、resident 参数、dequant workspace、block KV、Embedding、LM Head、full logits、临时空间和 CUDA 安全余量，并检查 MemAvailable、RLIMIT_MEMLOCK 与 `cudaMemGetInfo`。NUMA 检查仍待后续实现。

**修复方向：**增加 `preflight()`：估算 arena、slot、resident、KV、logits 的峰值，检查 `RLIMIT_MEMLOCK`、RAM 和 `cudaMemGetInfo`，失败时给出可执行配置。

#### 5.10 线程和资源生命周期不完整（核心对象与 soak 已修复第一版）

WeightStore、resident arena、runtime、vocab runtime、executor 和 KV manager 已实现 context manager 与幂等 `close()`；CLI 使用 `ExitStack` 做部分分配回滚；source Future 增加超时。`tools/soak_tiny.py` 覆盖累计 decode、重复加载、KV handle 循环、故障 runtime 重建和两种 WeightStore。soak 曾定位并修复新 CUDA stream 导致 cuBLAS workspace 按模型生命周期增长的问题。

**修复方向：**实现 `__enter__/__exit__`、关闭状态、Future cancel/timeout、信号处理和进程退出清理。

### P2：工程和产品化缺口

- 仅支持本地 safetensors，不支持 GGUF、PyTorch bin、远端流式读取或 SSD→DRAM 分层。
- 没有模型 tokenizer/config 版本锁定；checkpoint manifest 校验已实现。
- 没有统一日志、结构化 tracing、Prometheus 指标或可视化 profile 输出。
- 已有单元、合成 checkpoint、HF 数值、CUDA 端到端和本地真实 8B 回归；70B 真实 checkpoint 及外部框架公平基准仍未完成。
- 没有多 GPU、CPU NUMA 绑定、PCIe/NVLink 拓扑感知。
- 没有服务端 API、请求取消、超时、并发调度和安全限制。
- 生成 CLI 的默认路径仍指向历史 70B 目录；目录不存在时需显式传入 `CASCADE_CHAT_CHECKPOINT`。
- `return_full_logits` 适合验证但不适合生产，缺少强制的显存预算保护。

## 6. 需要优先补齐的开发顺序

### 阶段 A：正确性基线

1. [完成] 模型适配器和 config/index 自动校验。
2. [完成第一版] Hugging Face/PyTorch prefill、decode、逐阶段与 logits/Top-k 对比工具。
3. [完成] 预分配 block cache、最大长度、稳定地址、释放复用和 OOM 测试。
4. [进行中] 已恢复 plan、checkpoint、weight store、KV、MemoryPlanner、tiny Llama reference 测试；runtime failure/vocab/sampling 的扩充继续进行。

### 阶段 B：稳定流水线

1. [完成第一版] runtime 拆成 producer、H2D scheduler、compute consumer。
2. [部分完成] 已有有界队列和用户选择的固定 prefetch depth；动态 EWMA 水位待 M4。
3. [完成] 复用 CUDA event/device/staging slot，常规 decode 不再创建事件或 worker。
4. [完成第一版] full-pinned/staging preflight、worker/callback 错误传播、幂等 shutdown 和部分分配回滚。

### 阶段 C：真实性能

1. [完成第一版] 分别测量 pageable→pinned、联合 weight/scale H2D、dequant、GEMM、attention、LM Head。
2. [完成第一版] `tools/benchmark.py` 对 matrix/matrix_group/layer、两种 store 和两种 vocab placement 做版本化离线基准。
3. [W8A16 SM86 完成第一版] per-channel 已覆盖 prefill/decode；
   per-group 已覆盖 M=1 decode，prefill 必须显式选择 fallback。下一步是
   per-group scale mainloop 和 FusedW4A16。报告单独记录 provider 与
   fallback。
4. 评估 batch、prefill、decode 和长上下文四种负载，避免用单请求结果推断服务吞吐。

### 阶段 D：应用化

- paged KV cache + continuous batching；
- 请求级调度、取消、超时和 OpenAI-compatible API；
- SSD/DRAM/GPU 三级权重缓存；
- NUMA/PCIe 感知和多 GPU 分片；
- 可恢复 checkpoint 加载和完整运行报告。

## 7. 开发时必须遵守的接口约束

- `TransferUnit` 是当前最小可调度单位；unit 内所有 tensor 必须在 H2D 完成后才能被 callback 使用。
- callback 不得保存任何 device slot view；runtime 会在后续 unit 复用该 slot。
- 所有跨 stream 访问必须通过 CUDA Event 或显式 stream wait 建立 happens-before。
- CPU 原始权重是只读源，不允许把 GPU 结果写回覆盖 CPU arena。
- vocab streamed 模式只能返回 Top-k；只有测试场景才允许 materialize full logits。
- 新模型必须同时提供 geometry、权重 key、dtype/scale 规则和参考输出校验。

## 8. 最小验证命令

当前本地仓库未保留实验依赖和测试数据。修改源码后至少执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c \
  'import layer_streaming; print("framework imports: ok")'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
git diff --check
```

GPU 数值回归：

```bash
.venv/bin/python tools/generate_tiny_checkpoint.py \
  --output /tmp/cascade-tiny --quantization none --shards 2
.venv/bin/python tools/compare_reference.py \
  --checkpoint /tmp/cascade-tiny \
  --input-ids 1,13,7,22 --decode-ids 31,12
```

M5 stability 与 benchmark：

```bash
.venv/bin/python tools/soak_tiny.py \
  --checkpoint /tmp/cascade-tiny-int8 \
  --output /tmp/cascade-soak.json \
  --weight-store both --decode-tokens 1000 \
  --load-cycles 10 --cache-cycles 100

.venv/bin/python tools/benchmark.py \
  --checkpoint /tmp/cascade-tiny-int8 \
  --output /tmp/cascade-benchmark.json --preset full

.venv/bin/python tools/check_backends.py --device cuda:0
```

INT8/INT4 fallback 回归从 checkpoint metadata 读取格式：

```bash
.venv/bin/python tools/generate_tiny_checkpoint.py \
  --output /tmp/cascade-tiny-int8 \
  --quantization int8_per_group --group-size 64
.venv/bin/python tools/compare_reference.py \
  --checkpoint /tmp/cascade-tiny-int8 \
  --weight-format auto --profile
```

有模型和完整依赖后，再运行 `tools/run_llama31.py` 或聊天 CLI，并记录：冷加载时间、CPU 常驻字节数、每 token H2D、compute、source wait、GPU peak、KV cache bytes 和生成正确性。

## 9. 当前结论

当前代码已经进入核心接口冻结和可重复验证阶段。KV、checkpoint、多 dtype arena、资源生命周期、有界流水线、soak 与 benchmark 已有稳定边界。SM86 W8A16 ABI 2 已在 RTX 3080 Ti 完成现有真实闭环；INT8/INT4 fallback 仍必须显式选择。

NVIDIA 硬件兼容层已加入 HardwareProfile、RuntimeFeatureProfile、ProviderCapability、CompatibilityResolver、构建元数据和架构/ABI 隔离的 Numerical Contract Registry。`ExecutionPlan` schema v1 与 `RunReport` schema v2 没有改动。SM80、SM89、SM90 目前仅为 `declared/unqualified`，不得描述为已兼容；具体边界见 [NVIDIA_HARDWARE_COMPATIBILITY.md](NVIDIA_HARDWARE_COMPATIBILITY.md) 和 [COMPATIBILITY_MATRIX.md](COMPATIBILITY_MATRIX.md)。

Docker D1～D4 架构已落地：单一多阶段 Dockerfile、集中版本、统一 `cascade` CLI、Provider/Adapter/Quantizer entry point、qualified Bundle、固定挂载、非 root Compose 和无 GPU CI 契约均已建立。Core wheel 与 SM86 平台 Provider wheel 已完成离线结构验证。Generic 镜像实际拉取构建、容器 tiny smoke、SM86 Golden Suite 和 GHCR 发布仍属于 D5～D7，未完成前不得声明正式 Docker 镜像可用。详见 [DOCKER_GUIDE.md](DOCKER_GUIDE.md)。
