# Cascade-LLM KV Cache Architecture V2

本文档冻结 KV Cache V2 的正交策略接口，并记录当前代码的真实实现边界。
Paged KV 是唯一主线；精确、量化和稀疏是不同准确性等级，运行时不得静默跨级切换。

## 1. 正交策略

`layer_streaming.kv_policy.KVPolicy` 独立描述：

```text
accuracy   exact | quantized | sparse
storage    gpu | gpu_cpu | gpu_cpu_nvme
dtype      bf16 | fp16 | int8 | fp8 | int4
selection  none | quest_flat | hierarchical_quest | centroid_only
reuse      none | session | prefix_memory | prefix_persistent
page_size  16 | 32（正式 CLI）
```

当前 D1 可执行集合严格限制为：

```text
exact + gpu + (bf16 | fp16) + none + (none | session)
```

其他组合可以被 MemoryPlanner 做 metadata-only 预算，但执行前会返回明确的
`NotImplementedError`，不会回退到另一个准确性等级。

## 2. D1 物理结构

GPU K/V arena 使用：

```text
[layer, page, batch, kv_head, page_token, head_dim]
```

固定 layer 和 batch 后即为 block-major HND：

```text
[page, kv_head, page_token, head_dim]
```

每个请求拥有 `RequestBlockTable`，维护逻辑块到任意物理页的映射。页池按需分配，
不要求物理页连续，也不会因 Token append 扩大 GPU tensor。

已实现页面状态：

```text
FREE
ACTIVE_MUTABLE
SEALED_PRIVATE
SEALED_SHARED
OFFLOADING / RESTORING
CPU_RESIDENT / NVME_RESIDENT
EVICTING
```

其中 D1 只实际进入前四种状态；后五种用于后续 Tier 状态机，当前不能执行。

## 3. 生命周期规则

- Active partial page 不共享。
- 写满的页面变为 `SEALED_PRIVATE`。
- Fork 对完整 Sealed 页增加引用计数并变为 `SEALED_SHARED`。
- Fork 对未写满尾页立即复制，父子之后可独立追加。
- Attention 读取页面时增加 `pin_count`，结束后归还。
- pinned 或 transfer 状态页面不能释放。
- Reset/Release 逐页减少引用；归零后回到确定性空闲堆。
- Manager、Cache 和兼容 `SimpleKVCache` 均支持幂等关闭。

## 4. Dense Attention 后端

当前提供两个精确 Attention 语义的 reference backend：

| Backend | 默认 | 多页实现 | 完整 KV 临时区 | 当前用途 |
|---|---:|---|---:|---|
| `dense_paged_sdpa_reference` (`DensePagedAttention`) | 是 | 将当前 layer 的页面整理为连续 K/V 后调用 PyTorch SDPA | 有，随上下文增长 | 正确性基线 |
| `dense_paged_online_reference` (`DensePagedOnlineAttention`) | 否 | FP32 online-softmax 逐页读取 | 无 | 数据布局和 fused kernel 原型参考 |

两者的单页路径都使用 PyTorch SDPA 快路径，并支持 MHA/GQA、causal prefill 和
单 Token decode。默认选择 materialized SDPA，是因为它在当前真实 8B 多页验证中
保持了 Transformer 各阶段输出一致；online 路径虽然避免了完整 KV 临时区，但其
舍入路径与 BF16 SDPA 不同，真实 8B 误差会跨层传播，暂不能作为默认后端。

两个 backend 都不是生产性能实现。前者存在随上下文增长的 K/V copy 和 workspace，
后者包含 Python page loop 与大量 kernel launch。后续必须增加直接读取 Block Table 的
fused Dense Paged Attention Provider，才能同时关闭 D1 的内存、数值和性能门槛。

## 5. CLI

真实 Llama 入口支持：

```bash
python tools/run_llama31.py \
  --checkpoint MODEL \
  --kv-accuracy exact \
  --kv-storage gpu \
  --kv-dtype bf16 \
  --kv-index none \
  --kv-prefix-cache off \
  --kv-page-size 16
```

CLI 总是打印展开后的完整策略。以下请求当前会在 GPU 分配前拒绝：

```bash
--kv-accuracy quantized --kv-dtype int8
--kv-storage gpu-cpu
--kv-prefix-cache memory
--kv-accuracy sparse --kv-index quest-flat
```

它们分别属于 D4、D3/D2 和 D5，不会被模拟成 exact GPU dense。

## 6. MemoryPlanner 与 RunReport

MemoryPlanner 已独立输出：

```text
kv_gpu_pool_bytes
kv_cpu_pool_bytes
kv_nvme_budget_bytes
kv_index_bytes
kv_page_count
kv_page_bytes
kv_attention_workspace_bytes
```

RunReport schema 仍为 v2，没有修改冻结顶层字段；新增信息位于：

```text
runtime_config.kv_policy
timings.kv_attention_time_ms
pipeline.kv
memory.kv_gpu_pool_bytes
memory.kv_cpu_pool_bytes
memory.kv_nvme_budget_bytes
memory.kv_index_bytes
```

## 7. 当前里程碑状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| D0 | 已完成第一版 | 策略、布局、能力、内存计算、统计和连续 SDPA 消融基线 |
| D1 存储/生命周期 | 已完成第一版 | 非连续页表、Append/Free/Fork/COW/ref/pin、MHA/GQA reference |
| D1 数值门禁 | 部分完成 | SDPA reference 的 Transformer 多页阶段一致；online reference 尚未关闭误差传播 |
| D1 生产性能 | 未完成 | 缺 fused Dense Paged Attention；默认路径仍复制连续 K/V，且未达到短上下文 ≤10% 退化目标 |
| D2 | 未实现 | GPU Prefix Hash/LRU/Salt 尚未接入 |
| D3 | 未实现 | TransferArbiter 和 Pinned CPU Prefix Tier 尚未接入 |
| D4 | 未实现 | INT8 KV 页面和 fused Attention 尚未接入 |
| D5 | 未实现 | Quest Flat 尚未接入 |
| D6 | 未实现 | NVMe Persistent Prefix 尚未接入 |
| D7 | 未实现 | Hierarchical Quest 尚未接入 |
| D8 | 未实现 | Active KV Offload/SSD Decode 仍为研究项 |

详细数值和性能结果见 `KV_CACHE_D1_VALIDATION.md`。
