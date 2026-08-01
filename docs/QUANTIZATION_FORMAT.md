# Cascade-LLM 量化 checkpoint 格式

本文档定义当前合成 checkpoint 和 fallback backend 使用的物理格式。所有量化配置写入 `config.json.quantization_config`，运行时不得根据 tensor 内容猜测格式。

## 统一 metadata

```json
{
  "format": "cascade_symmetric",
  "bits": 4,
  "granularity": "per_group",
  "group_size": 64,
  "scale_dtype": "bfloat16",
  "packing": "int4_pair_uint8",
  "axis": 1,
  "execution_path": "int4_dequant_bf16_fallback"
}
```

当前执行 backend：

```text
bf16_linear
fp16_linear
int8_dequant_bf16_fallback
int8_dequant_fp16_fallback
int4_dequant_bf16_fallback
int4_dequant_fp16_fallback
```

带 `_fallback` 的路径先重建完整 A16 权重，再调用标准 `F.linear`。它们用于格式、数值和流水线验证，不代表 fused W8A8/W4A16 性能。

## INT8 symmetric

linear 逻辑 shape 均为 `[out_features, in_features]`，权重 key 为 `<linear>.weight`，scale key 为 `<linear>.weight_scale`，源码拼接形式是 `<linear>.weight + "_scale"`。

per-channel：

```text
weight  int8                 [out_features, in_features]
scale   bfloat16 / float16   [out_features, 1]

W[row, col] = A16(weight[row, col]) * scale[row, 0]
```

per-group（`group_size` 为 32/64/128，`axis=1`）：

```text
weight  int8                 [out_features, in_features]
scale   bfloat16 / float16   [out_features, in_features / group_size]

W[row, col] =
    A16(weight[row, col]) * scale[row, floor(col / group_size)]
```

当前只执行 symmetric INT8，不保存 zero-point。`in_features` 必须能被 group size 整除。

## packed INT4 symmetric

INT4 仅支持 per-group，scale shape 和分组公式同 INT8 per-group。逻辑矩阵按 row-major 展平后，每两个值打包进一个 `uint8`：

```text
low 4 bits   = 第一个值
high 4 bits  = 第二个值
stored 0..15 映射为 signed -8..7
physical shape = [ceil(out_features * in_features / 2)]
```

奇数个逻辑元素时，最后一个 byte 的高四位填 0；解包时根据逻辑元素数量丢弃 padding。当前合成量化器用 `max(abs(W))/7` 计算 scale 并生成 `[-7, 7]`，解包器仍完整支持 `-8`。

## 混合 dtype 和 tied weight

Transformer、Embedding、LM Head 和 Norm 的存储/计算 dtype 分别记录在 `WeightSpec` 中。常用组合：

```text
INT8 Transformer + BF16 Embedding/LM Head/Norm
INT4 Transformer + FP16 Embedding/LM Head + BF16 Norm
```

`tie_word_embeddings=true` 时 checkpoint 只保存 `model.embed_tokens.weight`；`lm_head.weight` 是 alias，不重复占用 CPU region 或 resident GPU arena。

## 启动校验

在分配 CPU/GPU 大内存前必须检查：

- bits、scheme、granularity、axis 和 group size；
- 逻辑 shape、packed storage shape 和 dtype；
- scale/zero-point key、shape 和 dtype；
- group size 整除关系及 256-byte backend alignment；
- 单文件或 index 中的 shard 存在性、重复 key、未索引 tensor；
- safetensors data offset 长度、越界和重叠；
- alias target 存在且不重复加载。

生成与回归：

```bash
.venv/bin/python tools/generate_tiny_checkpoint.py \
  --output /tmp/cascade-int4 \
  --layers 2 --hidden-size 256 --intermediate-size 768 \
  --attention-heads 8 --kv-heads 2 --vocab-size 1024 \
  --quantization int4_per_group --group-size 64 \
  --scale-dtype bf16 --dtype bf16 --shards 2

.venv/bin/python tools/compare_reference.py \
  --checkpoint /tmp/cascade-int4 --weight-format auto
```

生成目录的 `reference/model.safetensors` 保存未量化参考权重；数值工具则按 checkpoint 中实际量化值显式解量化，以隔离量化误差与执行器误差。

未来 fused backend 使用独立名称 `fused_w8a16`、`fused_w8a8` 或 `fused_w4a16`。在 kernel 注册前，选择这些名称必须明确报错，禁止静默切换到 fallback。

M5 起可通过 `ExecutionPolicy.linear_backend` 或 CLI `--backend` 明确选择。
真实 provider 必须实现冻结的 `LinearBackend` 并调用
`register_linear_backend()`；注册仅允许上述三个名称，且 storage、activation、
output dtype 必须与 frozen contract 一致。非 fallback 权重由 runtime 直接调用
`backend.execute()`，不能先构造完整 BF16/FP16 权重。
