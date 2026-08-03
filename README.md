# Cascade-LLM

Cascade-LLM 是面向单机单卡、单请求优先的 CPU 常驻权重流式推理框架。模型权重完整保存在 CPU 内存中，运行时按矩阵/矩阵组/层粒度异步执行 CPU→GPU H2D 与 GPU 计算，并通过固定资源池和预分配 block KV Cache 限制 GPU 占用。

当前重点是可验证的推理内核，不是通用推理服务器。统一执行计划由任意兼容 Llama `config.json` 和 checkpoint metadata 构建，支持 BF16、FP16、INT8 per-channel/per-group 与 packed INT4 per-group。INT8/INT4 fallback 始终显式命名；此外已提供可选的 SM86 CUTLASS W8A16 provider：per-channel 支持 prefill/decode，per-group 当前仅支持 M=1 decode，其他 fused 格式仍不会被静默模拟。

当前 KV 里程碑为 **KV Stack Beta**：generation-safe PagePool/Ownership、Fork/COW/Prefix/Beam/Speculative/rollback、Quest-flat CPU summary index、Mock GPU/CPU/SSD tiering、prefetch/coordinator 和显式 workload routing 已完成。SM86 CUDA 当前服务 Decode/Short Suffix；Full/Chunked Prefill 会记录原因并进入 `reference_paged_exact` correctness fallback。真实 CPU/NVMe/GDS 活动 KV、Quest 数据集质量和专用高吞吐 Prefill Kernel 尚未完成。权威状态见 `docs/KV_ARCHITECTURE_CONTRACT.md` 与 `docs/KV_REMEDIATION_IMPLEMENTATION_REPORT.md`。

## 迁移与 Codex 交接

迁移到新服务器时不要复制 `.venv`、模型进入 Git 或沿用旧 GPU 编译产物。使用以下入口：

- [`AGENTS.md`](AGENTS.md)：Codex 每次进入仓库自动读取的开发约定；
- [`docs/CODEX_HANDOFF.md`](docs/CODEX_HANDOFF.md)：项目目标、当前状态、暂停边界和大权重实验路线；
- [`docs/SERVER_MIGRATION.md`](docs/SERVER_MIGRATION.md)：Git/模型/依赖/Codex 的完整迁移步骤；
- `scripts/bootstrap_server.sh`：按锁文件创建 Python 3.10 环境；
- `scripts/verify_server.sh`：目标机 GPU、依赖和测试验收；
- `tools/capture_environment.py`：生成不包含凭据的机器环境回执。

## 核心代码

- `layer_streaming/adapter.py`：Llama config、RoPE/激活、显式执行策略和 WeightSpec 生成。
- `layer_streaming/specs.py`：逻辑/物理 shape、dtype、alias 和 QuantizationSpec。
- `layer_streaming/checkpoint.py`：单文件/多 shard safetensors manifest 与启动校验。
- `layer_streaming/execution_plan.py`：多 dtype byte offset、TransferUnit 和 workspace 计划。
- `layer_streaming/backends.py`：BF16/FP16、INT8/INT4 fallback 及 fused 后端占位接口。
- `layer_streaming/backend_capability.py`：SM/dtype/shape/alignment 能力查询及 prefill/decode backend sidecar。
- `layer_streaming/placement.py`：不修改冻结 plan schema 的静态整层 Transformer 常驻 sidecar。
- `layer_streaming/providers/cutlass/`：SM86 W8A16 CUTLASS per-channel 与专用 per-group M=1 decode provider。
- `layer_streaming/multi_dtype_store.py`：独立 BF16/FP16/INT8/INT4/scale CPU region。
- `layer_streaming/mixed_runtime.py`：通用 byte slot、双 CUDA stream 和 backend 分派。
- `layer_streaming/mixed_vocab.py`：混合 dtype Embedding/LM Head 流式执行。
- `layer_streaming/plan.py`：层、矩阵组和矩阵级显存计划。
- `layer_streaming/weight_store.py`：full-pinned 与 pinned-staging 权重存储。
- `layer_streaming/runtime.py`：H2D Copy Stream、Compute Stream、Event 和缓冲区生命周期。
- `layer_streaming/pipeline.py`：常驻 SourceProducer/H2DScheduler、ComputeConsumer 和三条有界队列。
- `layer_streaming/kv/`：Paged KV Runtime、请求表、页面所有权与 Prefix Cache；旧消融位于 `layer_streaming/experimental/`。
- `layer_streaming/memory.py`：RAM、memlock、GPU slots、KV、词表及 logits 的启动预检。
- `layer_streaming/reporting.py`：稳定的 `run_report.json` schema 和跨 token profile 聚合。
- `layer_streaming/stability.py`：长时间运行资源快照、趋势阈值和 stability report。
- `layer_streaming/benchmarking.py`：统一 benchmark schema、sample 和 median 聚合。
- `layer_streaming/llama31.py`：Llama 模型结构与流式 Transformer 执行。
- `layer_streaming/int8.py`：旧 INT8 per-channel BF16 兼容路径。
- `layer_streaming/vocab.py`：Embedding 按行读取、LM Head 词表分块与在线 Top-k。
- `layer_streaming/chat.py`：采样、停止词和会话状态辅助逻辑。
- `tools/run_llama31.py`：推理入口。
- `tools/qualify_backend.py`：真实矩阵族的 provider 数值和延迟资格测试。
- `tools/chat_llama31_70b_int8.py`：交互聊天入口。
- `scripts/chat_llama31_70b.sh`：聊天启动脚本。

## 运行

模型权重不提交到 Git。准备好兼容的 Llama 3.1 checkpoint 后：

下载用于真实回归的 BF16 Llama 3.1 8B checkpoint：

```bash
# 先检查镜像元数据、目标目录和磁盘空间，不下载权重。
bash scripts/download_llama31_8b.sh --dry-run

# 支持断点续传；脚本会在网络请求前清除大小写 proxy 环境变量。
bash scripts/download_llama31_8b.sh
```

默认从 `https://hf-mirror.com` 下载固定 revision 的
`unsloth/Meta-Llama-3.1-8B-Instruct` 标准 safetensors，约 15 GiB。可通过
`.env.local` 或命令行覆盖仓库、revision 和输出目录。若改用受限的 Meta
官方仓库，需要先接受其许可证并临时 `export HF_TOKEN=...`；不要将 token
写入 `.env.local`。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_llama31.py \
  --checkpoint /path/to/checkpoint \
  --weight-format auto \
  --backend checkpoint \
  --weight-store pinned_staging \
  --granularity matrix_group \
  --embedding-mode streamed \
  --lm-head-mode streamed \
  --slots 2 \
  --prefetch-depth 3 \
  --kv-block-size 16
```

聊天：

```bash
CASCADE_CHAT_CHECKPOINT=/path/to/checkpoint \
CASCADE_CHAT_GPU=0 bash scripts/chat_llama31_70b.sh
```

以下策略全部由用户显式选择，不会被运行时静默改写：

- `--granularity matrix|matrix_group|layer`：权重传输粒度；
- `--weight-format auto|bf16|fp16|int8_dequant_*|int4_dequant_*`：后端格式，`auto` 从 checkpoint metadata 读取；
- `--backend checkpoint|bf16_linear|fp16_linear|int8_dequant_*|int4_dequant_*|fused_*`：明确计算 backend；未注册 fused 直接失败；
- `--provider none|cutlass`：显式加载可选 provider；
- `--prefill-backend/--decode-backend`：显式阶段 backend；两阶段均在启动前资格检查；
- `--weight-store full_pinned|pinned_staging`：完整 CPU arena 是否锁页；
- `--embedding-dtype/--lm-head-dtype/--norm-dtype auto|bf16|fp16`：非 Transformer 权重 dtype；
- `--quant-granularity`、`--group-size`、`--scale-dtype`：量化布局；与 checkpoint 不一致会在 GPU 分配前失败；
- `--embedding-mode resident|streamed`：Embedding 是否常驻 GPU；
- `--lm-head-mode resident|streamed`：LM Head 是否常驻 GPU；
- `--gpu-resident-weight-budget 0|4GiB|8GiB|...`：按完整层前缀静态常驻 Transformer 权重；
- `--slots 1|2|3|4`：GPU/staging slot 数量；
- `--prefetch-depth 1..8`：source/ready 有界队列窗口；
- `--kv-block-size 16|32`：KV 分配块大小。

启动顺序固定为：config 解析 → checkpoint 全量 key/shape/dtype 校验 → 内存预检 → CPU arena 分配 → checkpoint 加载 → GPU 资源分配。任何预检冲突都会在加载大权重前失败。

`tools/run_llama31.py --output run_report.json` 默认记录 CUDA 分段时间；聊天入口可用 `--run-report run_report.json` 输出最近一轮。报告显式记录 weight format、用户策略、硬件、checkpoint/preflight 状态、H2D/compute/attention/MLP/dequant/LM Head 时间、队列等待/水位、吞吐及 CPU/pinned/GPU/KV 内存。

NVIDIA Compute Capability、Provider ABI、编译架构和资格状态可通过 `tools/inspect_hardware.py` 与 `tools/check_compatibility.py` 在启动前检查。当前完整真实验证边界仅为 SM86 RTX 3080 Ti；SM80、SM89、SM90 仍是未验证扩展骨架，详见 [硬件兼容文档](docs/NVIDIA_HARDWARE_COMPATIBILITY.md)。

## Docker 入口

容器架构提供稳定的 `cascade doctor/inspect/validate/run/chat/benchmark/qualify/quantize/shell` 命令。模型只读挂载到 `/models`，缓存和报告分别写入 `/cache`、`/results`：

```bash
export CASCADE_MODEL_DIR=/ssd/cascade-llm/models
./scripts/cascade-docker.sh build
./scripts/cascade-docker.sh doctor
./scripts/cascade-docker.sh run \
  --checkpoint /models/Llama-3.1-8B-Instruct \
  --backend bf16_linear --max-new-tokens 32
```

当前尚未发布 GHCR 正式镜像。Generic 镜像不包含 fused Provider；SM86/full 构建只有在资格二进制、ABI metadata 和 Numerical Contract 全部存在时才允许完成。详见 [Docker 使用指南](docs/DOCKER_GUIDE.md) 与 [插件开发契约](docs/PLUGIN_DEVELOPMENT.md)。

CUTLASS provider 的构建、支持矩阵、资格测试和许可要求见
[`docs/CUTLASS_PROVIDER.md`](docs/CUTLASS_PROVIDER.md)。
本机 M5.1 验证边界和未完成项见
[`docs/M5_1_VALIDATION.md`](docs/M5_1_VALIDATION.md)。
真实 8B 的 M6 常驻性能曲线、1000-token 稳定性和 fused 数值边界见
[`docs/M6_VALIDATION.md`](docs/M6_VALIDATION.md)。
Fused W8A16 与 fallback 的舍入差异、5 个失败 stage 和 golden contract
见 [`docs/FUSED_W8A16_NUMERICAL_INVESTIGATION.md`](docs/FUSED_W8A16_NUMERICAL_INVESTIGATION.md)。
普通用户的环境检查、冒烟生成、显存预算曲线和故障判定见
[`docs/USER_TESTING_GUIDE.md`](docs/USER_TESTING_GUIDE.md)。

## 正确性测试

运行无需 checkpoint/GPU 的单元测试和 tiny Llama CPU 回归：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

生成确定性 tiny checkpoint，并在 CUDA 上逐阶段对比显式解量化的 Hugging Face 参考：

```bash
.venv/bin/python tools/generate_tiny_checkpoint.py \
  --output /tmp/cascade-tiny \
  --layers 2 --hidden-size 256 --intermediate-size 768 \
  --attention-heads 8 --kv-heads 2 --vocab-size 1024 \
  --dtype bf16 --quantization none --shards 2

.venv/bin/python tools/compare_reference.py \
  --checkpoint /tmp/cascade-tiny \
  --input-ids 1,13,7,22 \
  --decode-ids 31,12
```

生成器还接受 `--embedding-dtype`、`--lm-head-dtype` 和 `--norm-dtype`，可直接构造 INT4 Transformer + FP16 词表 + BF16 Norm 等混合 checkpoint。

对比报告包含 Embedding、每层 Attention/MLP/hidden、final norm、logits 最大/平均误差，以及 Top-1/Top-k 一致性。

INT8/INT4 fallback 使用量化 checkpoint 和显式完整解量化参考：

```bash
.venv/bin/python tools/generate_tiny_checkpoint.py \
  --output /tmp/cascade-tiny-int8 \
  --quantization int8_per_group --group-size 64 --scale-dtype fp16

.venv/bin/python tools/compare_reference.py \
  --checkpoint /tmp/cascade-tiny-int8 --weight-format auto

.venv/bin/python tools/generate_tiny_checkpoint.py \
  --output /tmp/cascade-tiny-int4 \
  --quantization int4_per_group --group-size 64 --shards 2

.venv/bin/python tools/compare_reference.py \
  --checkpoint /tmp/cascade-tiny-int4 --weight-format auto
```

无需下载权重即可生成大模型的 64 位 offset、arena、shard 和 GPU 峰值计划：

```bash
.venv/bin/python tools/plan_metadata.py \
  --model-id synthetic-70b \
  --layers 80 --hidden-size 8192 --intermediate-size 28672 \
  --attention-heads 64 --kv-heads 8 --vocab-size 128256 \
  --weight-format int8_dequant_bf16_fallback \
  --quant-granularity per_group --group-size 128 \
  --max-context 4096 --output /tmp/cascade-70b-plan.json
```

M5 的接口冻结、soak、统一 benchmark、后端能力检查和真实 checkpoint 验收命令见
[`docs/API_STABILITY.md`](docs/API_STABILITY.md) 与
[`docs/M5_VALIDATION.md`](docs/M5_VALIDATION.md)。

## 环境

需要 Linux、NVIDIA CUDA、PyTorch 和 Transformers；`.env.example` 提供存储路径环境变量模板。

模型、基准数据、下载脚本和生成报告不属于运行时源码，不随仓库保存。
