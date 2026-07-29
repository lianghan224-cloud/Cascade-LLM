# Cascade-LLM

Cascade-LLM 是面向单机单卡的 CPU 常驻权重流式推理框架。模型权重完整保存在 CPU 内存中，运行时按矩阵/矩阵组/层粒度异步执行 CPU→GPU H2D 与 GPU 计算，并通过双缓冲、CUDA Event 和可复用显存 Arena 限制 GPU 权重占用。

## 核心代码

- `layer_streaming/weight_store.py`：full-pinned 与 pinned-staging 权重存储。
- `layer_streaming/plan.py`：层、矩阵组和矩阵级显存计划。
- `layer_streaming/runtime.py`：H2D Copy Stream、Compute Stream、Event 和缓冲区生命周期。
- `layer_streaming/llama31.py`：Llama 3.1 模型结构与流式 Transformer 执行。
- `layer_streaming/int8.py`：W8A8/INT8 权重与 scale 处理。
- `layer_streaming/vocab.py`：Embedding 按行读取、LM Head 词表分块与在线 Top-k。
- `layer_streaming/chat.py`：采样、停止词和会话状态辅助逻辑。
- `tools/run_llama31.py`：推理入口。
- `tools/chat_llama31_70b_int8.py`：交互聊天入口。
- `scripts/chat_llama31_70b.sh`：聊天启动脚本。

## 运行

模型权重不提交到 Git。准备好兼容的 Llama 3.1 checkpoint 后：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_llama31.py \
  --checkpoint /path/to/checkpoint \
  --weight-store full_pinned \
  --granularity matrix \
  --slots 2
```

聊天：

```bash
CASCADE_CHAT_CHECKPOINT=/path/to/checkpoint \
CASCADE_CHAT_GPU=0 bash scripts/chat_llama31_70b.sh
```

`full_pinned` 直接从 CPU 锁页权重异步传输；`pinned_staging` 在锁页内存有限时使用少量 staging slot。`matrix` 是默认的细粒度调度单位，也可选择 `matrix_group` 或 `layer`。

## 环境

需要 Linux、NVIDIA CUDA、PyTorch 和 Transformers；`.env.example` 提供存储路径环境变量模板。

模型、基准数据、下载脚本和生成报告不属于运行时源码，不随仓库保存。
