# Llama 3.1 8B 真实验证记录（2026-07-30）

## 结论

本轮使用本地 `unsloth/Meta-Llama-3.1-8B-Instruct` BF16
safetensors checkpoint 完成了真实结构、tokenizer、逐层数值和短序列性能验证。

- Cascade-LLM 可在单张 12 GiB RTX 3080 Ti 上执行该 8B BF16 模型。
- checkpoint 的 291 个 tensor、4 个 shard、key、shape 和 dtype 全部通过校验。
- Llama 3.1 RoPE scaling、128256 vocab 和 chat template 均被正确识别。
- 8-token prefill 和连续 2-token decode 共比较 297 个输出，全部通过。
- Embedding、Attention、MLP、hidden state 和 final norm 与 Hugging Face
  BF16 参考逐元素一致。
- 三次 logits 比较的最大绝对误差均为 0.0625，平均绝对误差为
  0.00174～0.00186，Top-1 和 Top-10 均完全一致。
- 69 个自动化测试全部通过。

## 环境

```text
GPU:          2 × NVIDIA GeForce RTX 3080 Ti 12 GiB, SM86
CPU RAM:      125 GiB
Python:       3.8.10
PyTorch:      2.4.1+cu121
Transformers: 4.45.2
Checkpoint:   16,060,522,496 bytes
```

本机未安装 vLLM、llama.cpp、TensorRT-LLM、AWQ、GPTQ 或 bitsandbytes，
因此当前“其他框架”基线仅使用 Hugging Face Transformers eager。

## 短序列结果

输入规模统一为 batch 1、prefill 8 token、decode 2 token。

| 项目 | Cascade 单卡 BF16 | HF Transformers BF16 |
|---|---:|---:|
| GPU 数量 | 1 | 2 |
| TTFT | 1457.8 ms | 288.5 ms |
| Decode | 1459.1 ms/token | 38.5 ms/token |
| GPU peak allocated | 459.5 MiB | 15.0 GiB（两卡合计） |
| 单张 12 GiB 可运行 | 是 | 否 |
| 权重位置 | CPU + pinned staging | 两卡 GPU 常驻 |

该表用于直接展示时间/显存取舍，不是同资源性能排名。HF BF16 权重本身约
14.96 GiB，无法装入单张 12 GiB GPU；实测基线把 0～13 层放在 GPU 0，
14～31 层、norm 和 LM Head 放在 GPU 1。Cascade 使用一张卡和 CPU
权重流式传输，因此 TTFT 约慢 5.1 倍、decode 约慢 37.9 倍，但 GPU
allocated 仅为 HF 两卡合计的约 1/33。

Cascade 双缓冲相对单 slot 的同一短测：

| 配置 | TTFT | Decode | GPU peak allocated |
|---|---:|---:|---:|
| slot=1 / prefetch=1 | 1818.6 ms | 1541.7 ms/token | 235.5 MiB |
| slot=2 / prefetch=2 | 1457.8 ms | 1459.1 ms/token | 459.5 MiB |

双缓冲将 TTFT 改善约 20%，decode 改善约 5%，代价是约 224 MiB
额外 GPU transfer slot。

## Cascade 瓶颈

双缓冲样本每个请求（prefill 加两次 decode）累计流式传输了约
39.0 GiB Transformer 权重。分项计时显示：

```text
pageable_to_pinned: 7242.0 ms（各异步事件累计，可与其他阶段重叠）
weight H2D:         2228.4 ms（各异步事件累计）
source wait:        3886.9 ms
compute wait:       3173.8 ms
GEMM:                 81.4 ms
```

当前真实 BF16 8B 的主要限制是 host staging 和权重到达等待，而不是
GEMM。后续优化应优先减少每 token 的 CPU copy、提高 producer/H2D
重叠，或使用真实 INT8/INT4 checkpoint 和 fused provider 降低传输量。

## memlock 说明

系统 `RLIMIT_MEMLOCK` 为 64 MiB，默认预检会拒绝 256～704 MiB 的
pinned staging 预算。实测 PyTorch CUDA pinned allocator 可成功分配
128 MiB，实际双缓冲运行也成功。测试使用显式
`--ignore-memlock-limit`，该选项只把 memlock 错误降为警告，不忽略
CPU RAM 或 GPU 显存预算错误。

## 复现命令

```bash
.venv/bin/python tools/accept_checkpoint.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B-Instruct \
  --output real_results/8b_bf16/acceptance_metadata.json \
  --metadata-only --backend checkpoint \
  --weight-store pinned_staging --vocab-mode streamed \
  --granularity matrix --slots 1 --ignore-memlock-limit

.venv/bin/python tools/compare_reference.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B-Instruct \
  --input-ids 128000,4,5,6,7,8,9,10 --decode-ids 4,5 \
  --device cuda:0 --reference-device-map balanced \
  --backend checkpoint --weight-store pinned_staging \
  --embedding-mode streamed --lm-head-mode streamed --slots 1 \
  --atol 0.05 --rtol 0.05 \
  --output real_results/8b_bf16/hf_balanced_vs_cascade.json

.venv/bin/python tools/benchmark.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B-Instruct \
  --output real_results/8b_bf16/cascade_double_buffer.json \
  --preset smoke --backend checkpoint \
  --prefill-tokens 8 --decode-tokens 2 \
  --warmup 1 --repeats 3 --slots 2 --prefetch-depth 2 \
  --vocab-chunk-mib 4 --device cuda:0 --ignore-memlock-limit

.venv/bin/python tools/benchmark_hf_reference.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B-Instruct \
  --output real_results/8b_bf16/hf_balanced_benchmark.json \
  --device-map balanced \
  --input-ids 1,5,6,7,8,9,10,11 --decode-ids 4,5

.venv/bin/python -m unittest discover -s tests -v
```

原始 JSON 位于 `real_results/8b_bf16/`。
