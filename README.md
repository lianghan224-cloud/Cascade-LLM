# Cascade-LLM

面向个人用户单机单卡场景的 CPU 常驻权重流式推理运行时。

Cascade-LLM 将完整 Llama-3.1-8B BF16 权重保存在 CPU 内存，只把当前
计算需要的矩阵组异步传入 GPU。Embedding 按 Token ID 查行，LM Head
沿词表方向分块并在线归并全局 Top-k，从而在消费级显卡上同时降低显存和
逐 Token 延迟。

> 当前发布分支：`llama-3.1-8b`
>
> 状态：研究原型；已在真实 Llama-3.1-8B 权重和 RTX 3080 Ti 上验证。

## 实测摘要

统一测试条件：

- 模型：Meta-Llama-3.1-8B，BF16，8,030,261,248 参数；
- GPU：NVIDIA RTX 3080 Ti 12 GB，PCIe Gen4 x16；
- CPU：AMD Threadripper 3970X，125 GiB RAM；
- batch=1，6-token prompt，1 次 decode 预热，7 次稳态 decode；
- 单卡运行，KV Cache 保存在 GPU；
- 显存指标为 `torch.cuda.max_memory_allocated`。

| 实现 | 中位延迟 | 速度 | CUDA 峰值显存 |
|---|---:|---:|---:|
| AirLLM 3.0.1，BF16，热分层文件 | 6555.016 ms/token | 0.153 token/s | 0.989 GiB |
| Cascade 上一版，词表常驻 GPU | 582.340 ms/token | 1.717 token/s | 2.197 GiB |
| **Cascade 当前版，词表流式化** | **624.162 ms/token** | **1.602 token/s** | **0.448 GiB** |

当前版本相对 AirLLM：

- 逐 Token 速度提升 **10.50 倍**；
- CUDA 峰值显存降低 **54.65%**，少用约 553 MiB；
- Prefill 从 7381.899 ms 降至 746.872 ms；
- H2D 有效吞吐从 11.137 GB/s 提高到约 24.04 GB/s。

当前版本相对 Cascade 上一版：

- CUDA 峰值显存从 2.197 GiB 降至 0.448 GiB，降低 **4.90 倍**；
- 节省 79.60% / 1.749 GiB；
- 保留 93.30% 吞吐，延迟增加 7.18%。

详细原始数据见：

- [`real_results/vocab_streaming_report.md`](real_results/vocab_streaming_report.md)
- [`real_results/vocab_streaming_summary.json`](real_results/vocab_streaming_summary.json)
- [`real_results/airllm/comparison_report.md`](real_results/airllm/comparison_report.md)

## 架构

```text
               CPU DRAM
    ┌────────────────────────────────┐
    │ 完整 BF16 权重 Arena           │
    │                                │
    │ Embedding [V,H]：按 Token 查行 │
    │ Transformer：按矩阵组连续布局  │
    │ LM Head [V,H]：按词表行分块    │
    └───────────────┬────────────────┘
                    │ H2D
                    ▼
                 GPU VRAM
    ┌────────────────────────────────┐
    │ Slot A：当前矩阵组             │
    │ Slot B：下一矩阵组预取         │
    │ Norm：小型常驻区               │
    │ KV Cache：随上下文增长         │
    └────────────────────────────────┘

       Copy Stream ∥ Compute Stream
```

### Transformer 矩阵组

每个 Decoder Layer 分成四个完整矩阵组：

| 顺序 | 矩阵组 | BF16 权重大小 |
|---:|---|---:|
| 1 | Q + K + V | 48 MiB |
| 2 | Attention O | 32 MiB |
| 3 | MLP Gate + Up | 224 MiB |
| 4 | MLP Down | 112 MiB |

最大矩阵组为 224 MiB，因此运行时预分配两个 224 MiB GPU Slot。当前版本
不切单个矩阵内部 Tile，也不使用自定义 CUDA 算子；组内计算仍调用标准
PyTorch `F.linear`、SDPA、RMSNorm 和逐元素算子。

### Embedding 按行传输

Embedding 权重保持 CPU 行连续布局。Prefill 只收集输入 Token 对应的行，
单 Token decode 只传：

```text
1 × 4096 × BF16 = 8 KiB
```

因此不需要在 GPU 上完整保存约 0.979 GiB Embedding。

### LM Head 词表分块

Llama-3.1-8B 的 LM Head 为 `[128256, 4096]`。运行时沿词表方向切成
约 128 MiB 的连续行块：

```text
每个完整块：16,384 行
分块数量：8
最后一块：13,568 行
```

每块完成 H2D 后调用标准 `F.linear` 计算局部 logits，再使用
`torch.topk` 将局部 Top-k 与当前全局候选归并。该过程不进行近似词表
裁剪，连续 8 个 greedy Token 已与 CPU 参考完全一致。

Llama-3.1-8B 的 `tie_word_embeddings=false`，因此 Embedding 和
LM Head 共享布局和调度代码，但保留 checkpoint 中两份不同的数值权重。

## 与 AirLLM 的架构区别

| 项目 | AirLLM | Cascade-LLM |
|---|---|---|
| CPU 权重 | 按层文件映射/page cache | 连续 CPU Arena |
| H2D 来源 | 实测为 pageable Tensor | full-pinned 或 pinned staging |
| GPU 权重分配 | Hook 动态安装、转 meta、释放 | 启动时预分配双 Slot |
| Transformer 粒度 | 完整层/模块 | 完整矩阵组 |
| Embedding | 完整模块装载 | 只传所需行 |
| LM Head | 完整模块装载 | 128 MiB 词表块 |
| 计算/传输重叠 | 主要预取下一层到 CPU | CUDA Copy/Compute 双流 |
| 模型通用性 | 较强 | 当前专门适配 Llama-3.1-8B |
| 锁页内存要求 | 较低 | full_pinned 约 14.958 GiB |

Cascade-LLM 选择模型专用执行器和更强的 CPU 内存约束，换取更低的 GPU
显存、更高 H2D 吞吐和稳定的单请求延迟。

## 支持范围

当前已支持：

- Llama-3.1-8B BF16 safetensors；
- batch=1、无 Padding 的 Prefill；
- 单 Token 自回归 Decode；
- GQA、RoPE、SDPA、RMSNorm 和 SwiGLU MLP；
- GPU 常驻 KV Cache；
- `full_pinned` 与 `pinned_staging` 两种 CPU 权重模式；
- `matrix`、`matrix_group` 和 `layer` 三种 Transformer 粒度；
- Embedding 按行提取；
- LM Head 词表分块和在线 Top-k；
- 单卡 CUDA 执行。

当前不支持或尚未优化：

- 多请求连续批处理和动态 Batch；
- Tensor Parallel / Pipeline Parallel；
- 单矩阵内部 Tile 流水；
- 自定义或融合 CUDA Kernel；
- INT8/INT4 权重传输；
- 分页、量化或 CPU Offload KV Cache；
- 通用 Hugging Face 模型自动适配；
- Windows 和 macOS。

## 环境要求

已验证环境：

- Ubuntu 20.04；
- Python 3.8；
- PyTorch 2.4.1 + CUDA 12.1；
- Transformers 4.45.2；
- 支持 BF16 的 NVIDIA GPU；
- 推荐至少 32 GiB 系统内存；
- `full_pinned` 需要能够锁定约 14.958 GiB CPU 内存；
- 模型目录和缓存建议保留至少 30 GiB SSD 空间。

12 GB 显卡不是硬性下限；短上下文权重峰值约 0.448 GiB，但实际需求还包括
CUDA Context、activation、workspace 和随上下文增长的 KV Cache。

Llama-3.1-8B BF16 KV Cache 约为 128 KiB/Token：

| 上下文长度 | KV Cache 估算 |
|---:|---:|
| 2K | 256 MiB |
| 4K | 512 MiB |
| 8K | 1 GiB |
| 32K | 4 GiB |

## 快速开始

### 1. 克隆发布分支

```bash
git clone --branch llama-3.1-8b \
  https://github.com/lianghan224-cloud/Cascade-LLM.git
cd Cascade-LLM
```

### 2. 创建 Python 环境

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# CUDA 12.1 PyTorch
.venv/bin/pip install \
  torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

.venv/bin/pip install -r requirements-bench.txt
```

验证 CUDA：

```bash
.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. 配置持久化目录

```bash
cp .env.example .env.local
```

默认模型目录为：

```text
/ssd/cascade-llm/models/Llama-3.1-8B
```

如需修改，请编辑 `.env.local`。该文件被 Git 忽略，不要在其中保存
Hugging Face Token 或其他凭据。

加载环境变量：

```bash
source scripts/activate_env.sh
```

### 4. 获取模型权重

模型权重不包含在本仓库中。请遵守 Meta Llama 3.1 License 和
Acceptable Use Policy。

通过 Hugging Face 官方受限仓库：

```bash
huggingface-cli login
.venv/bin/python scripts/download_llama31_8b.py
```

网络受限时，可使用固定 revision、逐文件 SHA-256 校验的 ModelScope
下载器：

```bash
.venv/bin/python scripts/download_llama31_8b_modelscope.py
```

下载器支持断点续传，不会把模型或访问凭据提交到 Git。

### 5. 检查环境

```bash
source scripts/activate_env.sh
.venv/bin/python scripts/check_real_environment.py
```

### 6. 运行生成

推荐的高性能配置：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_llama31.py \
  --checkpoint "${CASCADE_LLAMA31_8B}" \
  --weight-store full_pinned \
  --granularity matrix_group \
  --vocab-mode streamed \
  --top-k 10 \
  --prompt "The meaning of life is" \
  --max-new-tokens 8
```

锁页内存不足时：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_llama31.py \
  --checkpoint "${CASCADE_LLAMA31_8B}" \
  --weight-store pinned_staging \
  --granularity matrix_group \
  --vocab-mode streamed \
  --top-k 10 \
  --max-new-tokens 8
```

`pinned_staging` 减少锁页内存，但增加 CPU pageable→pinned 拷贝，速度会
明显低于 `full_pinned`。

## 复现实验

正式 7 次稳态 Decode 基准：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/real_llama31_benchmark.py \
  --checkpoint "${CASCADE_LLAMA31_8B}" \
  --weight-store full_pinned \
  --granularity matrix_group \
  --vocab-mode streamed \
  --top-k 10 \
  --slots 2 \
  --warmup-decode 1 \
  --decode-repeats 7 \
  --profile-repeats 3 \
  --output real_results/bench_full_pinned_matrix_group_vocab_streamed_s2.json
```

重新生成汇总：

```bash
.venv/bin/python benchmarks/summarize_vocab_streaming.py
```

运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

当前分支应通过 11 项单元测试。

真实模型实验的证据标准见
[`REAL_EXPERIMENT_PROTOCOL.md`](REAL_EXPERIMENT_PROTOCOL.md)。

## 正确性

当前版本与 Transformers CPU 参考连续比较 8 个 greedy Token：

- Token ID 8/8 完全一致；
- 完整 logits 形状为 `[8, 128256]`；
- 最低 cosine similarity 为 0.999739；
- 最大绝对误差为 0.265625；
- Top-10 overlap 为 9～10/10。

原始收据：

- [`real_results/correctness_vocab_streamed_matrix_group_8token.json`](real_results/correctness_vocab_streamed_matrix_group_8token.json)
- [`real_results/correctness_comparison_vocab_streamed_matrix_group_8token.json`](real_results/correctness_comparison_vocab_streamed_matrix_group_8token.json)

## 项目结构

```text
layer_streaming/
  plan.py          静态权重布局、矩阵组和词表分块计划
  weight_store.py  full_pinned / pinned_staging CPU Arena
  runtime.py       GPU双Slot、CUDA Stream和Event调度
  llama31.py       Llama-3.1-8B执行器与KV Cache
  vocab.py         Embedding按行和LM Head在线Top-k

tools/
  run_llama31.py   本地生成入口

benchmarks/
  real_llama31_benchmark.py       真实端到端基准
  real_llama31_correctness.py     CPU参考与流式正确性
  summarize_vocab_streaming.py    当前版本汇总
  airllm_llama31_8b_benchmark.py  AirLLM公平对照

real_results/
  vocab_streaming_report.md       当前实验报告
  vocab_streaming_summary.json    机器可读汇总
  airllm/                         AirLLM对照收据
```

## 设计取舍

Cascade-LLM 的目标不是替代通用推理框架，而是研究个人用户单机单请求下，
CPU→GPU 权重流式化能达到的显存和吞吐边界。

主要取舍：

- 用 14.958 GiB full-pinned CPU 内存换取约 24 GB/s H2D；
- 用模型专用执行器换取热路径无 Hook、无 meta 转换、无动态权重分配；
- 用 LM Head 每 Token 额外约 43.7 ms H2D，换取约 1.749 GiB GPU
  峰值节省；
- 暂不切矩阵内部 Tile，以避免开发自定义 GEMM。

## 贡献

欢迎通过 Issue 或 Pull Request 提交：

- 其他 Llama 尺寸适配；
- 长上下文 KV Cache 管理；
- 调度时间线和 Nsight Systems 分析；
- pinned staging 优化；
- INT8/INT4 传输；
- Linux/NVIDIA 环境复现报告。

提交性能数据时，请同时提供硬件、软件版本、checkpoint、测试 prompt、
warmup、样本数量、原始 JSON 和正确性结果。合成基准不得标记为真实模型
端到端结果。

## 致谢

项目思路来源于 [AirLLM](https://github.com/lyogavin/airllm)。感谢
PyTorch、Hugging Face Transformers、safetensors 和 Meta Llama 社区。

## 许可证与模型条款

本仓库当前尚未包含代码开源许可证。维护者选择并提交正式 `LICENSE`
之前，请不要假设代码已获得再分发或商用授权。

Llama-3.1-8B 模型权重不属于本项目许可证范围。模型下载和使用必须遵守
Meta Llama 3.1 License、Acceptable Use Policy 以及模型分发平台条款。
