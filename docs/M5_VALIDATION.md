# M5 稳定性、Benchmark 与后端验收

## 1. 长时间稳定性

`tools/soak_tiny.py` 复用同一套推理路径，记录：

```text
CUDA allocated/reserved
owned pinned/pageable bytes
总线程与 Cascade worker 数
source/ready/free/staging queue 深度和容量
CUDA event 数量
KV total/allocated blocks 和 active handles
采样 token 延迟及首尾窗口趋势
```

1000 token、10 次 KV 生命周期、两种 CPU 模式：

```bash
.venv/bin/python tools/soak_tiny.py \
  --checkpoint /tmp/cascade-tiny-int8 \
  --output /tmp/cascade-soak-1k.json \
  --weight-store both \
  --decode-tokens 1000 \
  --load-cycles 10 \
  --cache-cycles 100 \
  --sample-every 10 \
  --fault-recovery-cycles 1
```

完整 100 次加载和 10000 token：

```bash
.venv/bin/python tools/soak_tiny.py \
  --checkpoint /tmp/cascade-tiny-int8 \
  --output /tmp/cascade-soak-10k.json \
  --weight-store both \
  --decode-tokens 10000 \
  --load-cycles 100 \
  --cache-cycles 1000 \
  --sample-every 25
```

工具会故意让 compute callback 失败，验证异常传播、幂等关闭和 runtime
重建。worker timeout、source worker 异常和队列背压另由
`test_runtime_failure.py` 与 `test_runtime_order.py` 做确定性故障注入。

M5 烟测曾发现“每次 runtime 重建新增约 8.1 MiB cuBLAS stream workspace”
的阶梯增长。当前按 CUDA device 复用 Copy/Compute stream 对，保留双流语义，
同时使重复加载后的 CUDA allocated/reserved 稳定。

## 2. 统一 Benchmark

固定入口：

```bash
.venv/bin/python tools/benchmark.py \
  --checkpoint /tmp/cascade-tiny-bf16 \
  --checkpoint /tmp/cascade-tiny-fp16 \
  --checkpoint /tmp/cascade-tiny-int8 \
  --checkpoint /tmp/cascade-tiny-int4 \
  --output /tmp/cascade-benchmark.json \
  --preset full \
  --backend checkpoint
```

`full` 覆盖：

```text
matrix / matrix_group / layer
full_pinned / pinned_staging
resident vocab / streamed vocab
```

每个 checkpoint 从 metadata 确定存储格式。也可通过 `--backend` 明确要求
某个 backend；实际计划与请求不一致会成为 error，不会 fallback。

每个 case 保存所有 sample 和 median：

```text
pageable_to_pinned_ms
combined_weight_scale_h2d_ms
weight_h2d_bytes / scale_h2d_bytes
dequant_ms / gemm_ms / attention_ms
embedding_ms / lm_head_ms
source_wait_ms / compute_wait_ms
ttft_ms / decode_ms_per_token
GPU peak allocated/reserved
queue max depth
actual_backends / fallback_backends
```

Weight 与 scale 当前在同一个 TransferUnit copy 中提交。为避免通过额外 copy
改变被测调度，报告只给出联合 H2D CUDA event 时间，并分别给出字节数，不伪造
两段独立耗时。

合成 benchmark 只用于回归和策略比较，不能作为真实模型性能结论。

## 3. Fused provider

能力检查：

```bash
.venv/bin/python tools/check_backends.py \
  --device cuda:0 \
  --output /tmp/cascade-backends.json

.venv/bin/python tools/check_backends.py \
  --device cuda:0 \
  --require fused_w8a16
```

SM86 的第一版真实 W8A16 provider 已落地。per-channel 的 `M>=1`（包括
M=1 decode）使用 CUTLASS Ampere mixed-input GEMM；per-group M=1 使用专用
CUDA weight-only GEMV。两者直接读取 INT8 weight，`workspace_bytes=0`。
per-group group size 32/64/128 可用；BF16/FP16 activation 和 BF16/FP16
scale 均通过资格测试。

```bash
.venv/bin/python tools/check_backends.py \
  --device cuda:0 --provider cutlass --require fused_w8a16

.venv/bin/python tools/qualify_backend.py \
  --backend fused_w8a16 --provider cutlass \
  --preset 8b --matrix all --m 1 2 4 8 16 32 128 512
```

资格工具会先检查 SM、dtype、量化粒度、group size、M/N/K 和 alignment，
再运行数值与延迟测试。`fused_w4a16` 和 `fused_w8a8` 仍为 unavailable；
W8A16 per-group 的 M>1 会明确拒绝，不会自动 fallback。构建、支持矩阵、
阶段 backend sidecar 和许可信息见 `docs/CUTLASS_PROVIDER.md`。

## 4. 真实 checkpoint 验收

不自动下载模型。对本地小型 Llama-family checkpoint：

```bash
.venv/bin/python tools/accept_checkpoint.py \
  --checkpoint /path/to/local-llama \
  --output /tmp/cascade-acceptance.json \
  --backend checkpoint \
  --weight-store pinned_staging \
  --vocab-mode streamed
```

入口检查 config geometry、真实权重命名、单/多 shard、dtype/shape、tokenizer
大小和特殊 token、内存预检，并在 CUDA 上与 Hugging Face 比较 prefill、
decode、逐层输出、logits 和 Top-k。`--metadata-only` 可只执行启动前检查。

报告中的 `synthetic` 字段必须为 `false` 才能计入 M5E 真实模型验收。合成
checkpoint 通过只证明工具和框架路径可运行。
