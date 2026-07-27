# Cascade-LLM

面向个人用户单机单卡场景的 CPU 常驻权重流式推理研究原型。项目思路来源于
[AirLLM](https://github.com/lyogavin/airllm)，当前分支在真实
Llama-3.1-70B-Instruct W8A8 checkpoint 上验证细粒度 H2D、双缓冲和
CPU 锁页内存策略。

> 当前分支：`llama-3.1-70b-int8`
>
> 8B 稳定实验保留在 `llama-3.1-8b` 分支。

## 真实实验结果

测试环境：

- 模型：`RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w8a8`；
- 固定 revision：`8d0dcbba33eeef589b0a607e46abe05a5a6431a8`；
- 15 个 safetensors，1,283 个 tensor，精确载荷 72,669,806,592 bytes；
- 560 个 INT8 线性权重，723 个 BF16 scale、norm、Embedding 和 LM Head；
- GPU：单张 NVIDIA RTX 3080 Ti 12 GiB；
- CPU：AMD Threadripper 3970X，125 GiB RAM；
- PyTorch 2.4.1+cu121，Transformers 4.45.2；
- batch=1，6-token prompt，单 token 自回归 decode。

| CPU 权重模式 | Transformer 粒度 | Slot | 中位延迟 | 速度 | H2D | CUDA 峰值显存 | 锁页 CPU 内存 |
|---|---|---:|---:|---:|---:|---:|---:|
| full_pinned | 矩阵 | 1 | 3680.545 ms | 0.272 token/s | 24.052 GB/s | 0.670 GiB | 67.684 GiB |
| **full_pinned** | **矩阵** | **2** | **2965.803 ms** | **0.337 token/s** | **24.064 GB/s** | **1.327 GiB** | **67.684 GiB** |
| full_pinned | 层 | 2 | 2985.189 ms | 0.335 token/s | 24.085 GB/s | 2.483 GiB | 67.684 GiB |
| pinned_staging | 矩阵 | 1 | 7268.460 ms | 0.138 token/s | 21.536 GB/s | 0.670 GiB | 0.348 GiB |
| pinned_staging | 矩阵 | 2 | 6190.552 ms | 0.162 token/s | 16.843 GB/s | 1.326 GiB | 0.692 GiB |
| pinned_staging | 矩阵组 | 2 | 6815.273 ms | 0.147 token/s | 18.742 GB/s | 1.764 GiB | 1.129 GiB |
| **pinned_staging** | **层** | **2** | **5695.903 ms** | **0.176 token/s** | **12.644 GB/s** | **2.483 GiB** | **1.848 GiB** |

推荐配置：

- 专用推理机器：`full_pinned + matrix + slots=2`；
- 无法锁定约 67.7 GiB 内存：`pinned_staging + layer + slots=2`。

主配置的关键结论：

- 相对同粒度单缓冲提速 1.241×；
- 实测 CUDA 峰值权重路径显存 1.327 GiB；
- 相对完整 67.679 GiB checkpoint tensor 载荷缩减 51.01×，节省 98.04%；
- 每 token 传输 70.566 GB 权重；
- Transformer H2D 为 2845.051 ms，有效吞吐 24.064 GB/s；
- Transformer compute 为 680.513 ms，其中 GPU BF16 解量化 489.726 ms；
- Transformer 与 LM Head 的 H2D event 总和为 2932.486 ms，占 token wall
  的 98.88%；
- 双缓冲隐藏了约 96.65% 的可重叠计算时间。

完整 70B INT8 权重无法装入 12 GiB GPU，因此没有伪造“普通 full-GPU”
速度基线。51.01× 是精确 checkpoint 载荷与实测流式 CUDA peak 的显存口径
对比，不是 full-GPU 端到端速度倍数。

详细报告和原始收据：

- [`real_results/70b_int8/report.html`](real_results/70b_int8/report.html)
- [`real_results/70b_int8/summary.json`](real_results/70b_int8/summary.json)
- [`real_results/70b_int8/checkpoint_validation.json`](real_results/70b_int8/checkpoint_validation.json)
- [`real_results/70b_int8/`](real_results/70b_int8/)

## 架构

```text
                         CPU DRAM
  ┌──────────────────────────────────────────────────────┐
  │ 72.67 GB 混合 dtype checkpoint arena                 │
  │                                                      │
  │ full_pinned：按传输单元边界拆成多个锁页 chunk         │
  │ pinned_staging：pageable 全量 arena + 有限锁页 slot   │
  └─────────────────────────┬────────────────────────────┘
                            │ INT8/BF16 H2D
                            ▼
                          GPU VRAM
  ┌──────────────────────────────────────────────────────┐
  │ Raw Slot A/B：checkpoint 原始 INT8 权重              │
  │ BF16 Workspace A/B：按矩阵复用的解量化输出           │
  │ 小型常驻区：norm 与 scale                            │
  │ Embedding：只传当前 token 行                         │
  │ LM Head：沿词表分块传输并在线合并 Top-k              │
  └──────────────────────────────────────────────────────┘

           H2D Copy Stream  ∥  Compute Stream
```

### Transformer

最低调度粒度为一个完整矩阵，不切矩阵内部 Tile，也不开发自定义 GEMM。
70B 模型共有 80 个 Decoder Layer；每层 7 个线性矩阵，因此矩阵粒度每
token 有 560 个 Transformer 传输单元。

运行时直接传输 checkpoint 中的 INT8 矩阵和 BF16 per-output-channel
scale，在 GPU 的可复用 workspace 中执行：

```text
BF16_weight = INT8_weight × BF16_scale
```

随后调用标准 PyTorch BF16 `F.linear`、SDPA、RMSNorm、RoPE 和 SwiGLU。
这能验证 CPU 常驻权重与 H2D 流水，但不是最终的 W8A8 activation-quantized
kernel。

### Embedding 和 LM Head

- Embedding 为 BF16 `[V,H]`，按输入 token ID 只提取并传输需要的行；
- LM Head 为 BF16 `[V,H]`，按 8,192 行、约 128 MiB 分成 16 块；
- 每块计算局部 logits，再在线维护全局 Top-k；
- 不需要让约 1.96 GiB 的完整词表矩阵常驻 GPU。

## 为什么两种模式的最优粒度不同

`full_pinned` 的 CPU 权重可以直接异步 H2D。层粒度和矩阵粒度都能跑满约
24 GB/s，矩阵粒度还快 0.65%，并少用约 1.16 GiB GPU 显存，因此选
矩阵。

`pinned_staging` 需要先做 pageable DRAM→锁页 slot 的 CPU copy。细分为
560 个矩阵后，staging 生产者频繁同步并与 H2D 竞争内存带宽；本机实测层
粒度虽然 H2D event 吞吐更低，但端到端 source wait 更少，最终比矩阵双缓冲
快 8.73%。因此“粒度越细越快”并不成立。

## 环境

建议：

- Linux + NVIDIA CUDA；
- Python 3.8+；
- PyTorch 2.4.1 + CUDA 12.1；
- Transformers 4.45.2；
- 至少 90 GiB 可用系统内存；
- 至少 85 GiB SSD 空间保存 checkpoint；
- `full_pinned` 需要约 67.7 GiB 可锁页内存和足够高的 memlock 限额；
- `pinned_staging` 不要求锁定完整权重。

创建环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements-bench.txt
```

## 获取模型

模型权重不在 Git 仓库中。国内镜像、清除代理、固定 revision、断点续传：

```bash
bash scripts/download_llama31_70b_int8_cn.sh
```

可以覆盖目标目录或并发数：

```bash
MODEL_DIR=/ssd/cascade-llm/models/Llama-3.1-70B-Instruct-W8A8 \
DOWNLOAD_WORKERS=4 \
bash scripts/download_llama31_70b_int8_cn.sh
```

中断后运行同一命令即可继续，不要删除 `.cache` 或 `.incomplete` 文件。

## 运行真实实验

先验证 checkpoint：

```bash
.venv/bin/python benchmarks/validate_llama31_70b_int8_checkpoint.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-70B-Instruct-W8A8 \
  --output real_results/70b_int8/checkpoint_validation.json
```

推荐的 full-pinned 矩阵双缓冲：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/real_llama31_70b_int8_benchmark.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-70B-Instruct-W8A8 \
  --weight-store full_pinned \
  --granularity matrix \
  --slots 2 \
  --decode-repeats 3 \
  --profile-repeats 2 \
  --output real_results/70b_int8/bench_full_pinned_matrix_s2_repeat.json
```

低锁页内存的 staging 层双缓冲：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/real_llama31_70b_int8_benchmark.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-70B-Instruct-W8A8 \
  --weight-store pinned_staging \
  --granularity layer \
  --slots 2 \
  --decode-repeats 3 \
  --profile-repeats 2 \
  --output real_results/70b_int8/bench_pinned_staging_layer_s2_repeat.json
```

加载阶段会真实读取约 72.67 GB 权重到 CPU。推理测量窗口内权重从 CPU
DRAM 读取并传往 GPU，不是从 SSD 逐层读取；重复实验的 decode I/O 增量仅
为少量系统元数据读取。

重新汇总和生成报告：

```bash
.venv/bin/python benchmarks/summarize_llama31_70b_int8.py
.venv/bin/python benchmarks/build_llama31_70b_int8_artifact.py
```

运行单元测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 正确性边界

已经验证：

- checkpoint key、shape、dtype 和总字节数全部匹配；
- `full_pinned` 与 `pinned_staging` 推荐模式连续 6 步生成相同 token；
- 两种模式每一步的 Top-10 完全相同；
- 生成文本为 ` not just about technology, but`。

尚未验证：

- 当前环境没有安装官方 `compressed_tensors` 执行参考；
- 尚未做本运行时与官方 W8A8 kernel 的逐层 activation/logits 对比；
- 因此不能宣称模型级完整数值等价；
- 尚未测长上下文、批处理、多请求和多 GPU。

## 项目结构

```text
layer_streaming/
  int8.py          70B 混合 dtype 计划、CPU store、GPU slot 与解量化流水
  llama31.py       通用 Llama-3.1 decode 执行器与 KV cache
  vocab.py         Embedding 按行和 LM Head 分块在线 Top-k

benchmarks/
  real_llama31_70b_int8_benchmark.py       真实端到端实验
  validate_llama31_70b_int8_checkpoint.py  checkpoint 完整性校验
  summarize_llama31_70b_int8.py            汇总与 QA
  build_llama31_70b_int8_artifact.py       可移植报告数据构建

real_results/70b_int8/
  bench_*.json                原始实验收据
  checkpoint_validation.json  checkpoint 校验
  summary.json                机器可读汇总
  report.html                 可移植详细报告
```

## 设计取舍

- 用约 67.7 GiB full-pinned CPU 内存换取约 24 GB/s H2D；
- 用两个 raw INT8 slot 和两个 BF16 workspace 换取传输/计算重叠；
- 用矩阵粒度节省显存，但不在矩阵内部切 Tile；
- 用模型专用执行器换取无 Hook、无 meta 转换、热路径无权重分配；
- 暂不将当前 BF16 解量化执行路径包装成“原生 W8A8 性能”。

下一阶段重点是官方参考正确性、融合 INT8/W8A8 kernel、KV cache 分页和
长上下文显存管理。

## 许可证与模型条款

本仓库当前尚未包含代码开源许可证。维护者提交正式 `LICENSE` 前，请不要
假设代码已获得再分发或商用授权。

模型权重不属于本仓库许可证范围。下载与使用必须遵守 Meta Llama 3.1
License、Acceptable Use Policy 和模型分发平台条款。
