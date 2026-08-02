# KV Cache V2 D0/D1 验证与消融报告

## 1. 验证范围

本轮只验收 D0 和 D1 reference，不宣称 D2～D8 已实现，也不把 synthetic
microbenchmark 当作真实 8B 性能结论。

环境：

```text
GPU: NVIDIA GeForce RTX 3080 Ti
Compute Capability: 8.6
PyTorch: 2.4.1+cu121
CUDA Runtime: 12.1
```

原始结果：

```text
real_results/kv_d1/ablation_sm86_reference.json
real_results/kv_d1/real8b_golden_single_page.json
real_results/kv_d1/real8b_golden_multipage_sdpa.json
real_results/kv_d1/real8b_golden_multipage_online_failure.json
```

复现命令：

```bash
.venv/bin/python tools/ablate_kv_cache.py \
  --device cuda:0 \
  --contexts 16,32,128,512 \
  --page-sizes 16,32 \
  --phase both \
  --attention-heads 8 \
  --kv-heads 2 \
  --head-dim 128 \
  --paged-backend both \
  --warmup 2 \
  --repeats 5 \
  --output real_results/kv_d1/ablation_sm86_reference.json
```

## 2. 自动化回归

```text
```text
120 tests passed
```
```

覆盖：

- 非连续物理页映射。
- Page append、reset、release 和 100 次复用。
- Sealed 页共享、引用计数、Fork、Partial Tail Copy-on-Write。
- MHA/GQA prefill 与 decode。
- MemoryPlanner 的 BF16/INT8、CPU/NVMe budget 和 Quest index 预算。
- RunReport v2 的 KV policy/profile 序列化。
- tiny BF16/FP16/INT8/INT4 checkpoint 端到端回归。

## 3. 数值消融

### Synthetic 单页

两个 backend 的单页都走 SDPA 快路径，16-token 和 32-token 配置对连续 SDPA
输出 bitwise 一致。

### Synthetic 多页

默认 `dense_paged_sdpa_reference` 整理页面后调用相同的 PyTorch SDPA 数值路径，
16/16 个 synthetic case 均通过 `atol=4e-3, rtol=4e-3` 门禁。

实验性 `dense_paged_online_reference` 以 FP32 online-softmax 逐页归并，15/16
个 case 通过同一门禁。它与 BF16 SDPA 的舍入路径不同，但在代表性多页 case 中，
相对 FP32 dense reference 的平均误差反而更小：

| Phase | Context/Page | 连续 BF16→FP32 mean error | Paged→FP32 mean error |
|---|---:|---:|---:|
| Prefill | 32/16 | 0.0005669 | 0.0004324 |
| Decode | 32/16 | 0.0004429 | 0.0002959 |
| Prefill | 128/16 | 0.0003533 | 0.0002555 |
| Decode | 128/16 | 0.0002125 | 0.0001480 |
| Prefill | 512/16 | 0.0002086 | 0.0001467 |
| Decode | 512/16 | 0.0001166 | 0.0000796 |

online backend 的缺陷：32/16 prefill 对 BF16 SDPA 的 pairwise max absolute error 为
`0.015625`，没有通过临时 `atol=4e-3, rtol=4e-3` pairwise gate；但该 case
中 online 路径对 FP32 reference 的 max/mean error 均小于连续 BF16 SDPA。
因此不能通过单纯放宽 pairwise 容差处理，后续 Numerical Contract 应同时记录：

1. BF16 SDPA pairwise error；
2. FP32 dense reference error；
3. 逐层 hidden/logits 误差；
4. Greedy/Top-k 一致率。

真实 tiny INT4 端到端回归中，切换 Paged reference 后实测：

```text
prefill logits max_abs_error = 0.001953125
decode logits max_abs_error  = 0.00146484375
```

该门禁固定为 `atol=2e-3, rtol=2e-3`，不是依据失败结果继续放宽。

### 真实 Llama-3.1-8B 单页 Golden

使用真实 BF16 checkpoint、8-token prefill 和两步 decode，与 Hugging Face
双卡常驻参考比较，共 `297` 个逐阶段项目：

```text
297 / 297 passed
prefill Top-1: equal
decode step 0 Top-1: equal
decode step 1 Top-1: equal
prefill logits max/mean: 0.0625 / 0.0017664
decode 0 logits max/mean: 0.0625 / 0.0018590
decode 1 logits max/mean: 0.0625 / 0.0017387
```

调查中曾发现单页 Decode 传入全 True attention mask 会让 CUDA SDPA 选择不同的
kernel/舍入路径，最终虽然 Top-1 一致，但旧门禁失败。修复为单 Token 尾部 Decode
使用无 mask、`is_causal=false` 后，同一 Golden 全部通过。回归测试现在明确检查该
调用不携带 `attn_mask`。

该结果只覆盖单页（总长度 10 < page size 16），不能代替多页 4K～128K Golden。

### 真实 Llama-3.1-8B 多页 Golden

17-token prefill 跨越 page-size 16 的边界，再执行两步 decode：

| Backend | 逐阶段通过 | Transformer 阶段 | Logits/Top-k | 结论 |
|---|---:|---|---|---|
| materialized SDPA | 295/297 | embedding、32 层 attention/MLP/hidden 和 final norm 全部零误差 | prefill 和 decode-1 各有一个门禁失败 | KV/Transformer 基线通过，差异定位到后续 streamed LM Head |
| online FP32 | 273/297 | 24 项失败，误差在后层 MLP/hidden/final norm 继续传播 | 三个 logits 均失败，但 Top-1 均一致 | 不能作为默认数值路径 |

materialized SDPA 的三个 logits max/mean error 分别为：

```text
prefill: 0.1250 / 0.0038592, Top-1 不同, Top-10 set consistency 1.0
decode 0: 0.0625 / 0.0026602, Top-1 相同, Top-10 set consistency 1.0
decode 1: 0.0625 / 0.0029286, Top-1 相同, Top-10 set consistency 0.9
```

由于对应 `final_norm` 比较为零误差，这两项整体门禁失败已经定位在 LM Head 及其
streamed reduction 路径，不是 Paged KV 或 Transformer Attention。它仍需单独关闭，
不能将整个真实 8B 报告写成通过。

online 路径的三个 logits max error 为 `0.46484375 / 0.23681640625 /
0.28125`（prefill/decode-0/decode-1）。这证明 synthetic 中“相对 FP32 更准确”
不足以建立整模型 Numerical Contract；当前不能通过放宽容差掩盖误差传播。

## 4. 性能消融

以下为 backend median / 连续 SDPA median。每格为 `materialized SDPA / online`，
越接近 1 越好：

| Phase | Context | Page 16 | Page 32 |
|---|---:|---:|---:|
| Prefill | 16 | 3.44× / 3.38× | 3.54× / 3.53× |
| Decode | 16 | 3.50× / 3.45× | 3.57× / 3.52× |
| Prefill | 32 | 5.71× / 23.36× | 3.59× / 3.46× |
| Decode | 32 | 5.81× / 23.30× | 3.49× / 3.52× |
| Prefill | 128 | 13.27× / 76.91× | 8.24× / 42.24× |
| Decode | 128 | 11.35× / 66.13× | 7.11× / 35.29× |
| Prefill | 512 | 26.84× / 181.65× | 14.55× / 92.61× |
| Decode | 512 | 34.80× / 232.56× | 19.09× / 119.79× |

结论：两个 reference backend 都明显不满足“短上下文端到端性能下降不超过 10%”。
materialized SDPA 较快，但代价是随上下文增长的连续 K/V copy；online 路径没有该
临时区，却受逐页 Python dispatch、多个 matmul/softmax launch 和在线归并拖累。
Page 32 在多页时通常更快，但不能据此直接设为默认值，因为还需同时比较内存碎片、
Prefix 命中粒度和后续 Quest 索引开销。

## 5. 内存消融

- Paged arena 在启动时一次预分配，Append 不改变 keys/values 地址。
- 默认 materialized SDPA 每次多页 Attention 会构造当前层连续 K/V，并因 GQA
  `repeat_interleave` 产生额外临时区；MemoryPlanner 已将
  `kv_attention_workspace_bytes` 计入 GPU 峰值预算。
- online reference 的 `materialized_full_kv_bytes = 0`，但尚未通过真实 8B 数值门禁，
  且性能最差，不能据此宣称生产路径已经消除完整 KV 临时区。
- Context 不能整除 page size 时，仅产生最后一页的固定 rounding overhead。
- Fork 只复制未写满尾页；完整 Sealed 页只增加引用计数。
- MemoryPlanner 中 INT8 KV 页池预算严格为 BF16 的一半。

## 6. 已完善部分

- 正交策略和准确性级别拒绝规则。
- HND page pool 与非连续 request block table。
- Append/Free/Reset/Fork/COW/ref_count/pin_count。
- MHA/GQA exact 语义 reference，以及 correctness-first 默认 SDPA 路径。
- GPU page pool、CPU/NVMe budget 和 index metadata 预算。
- CLI 展开、RunReport 埋点和可复现消融工具。

## 7. 仍有缺陷或未完成部分

### P0

1. 缺直接读取 Block Table 的 fused Dense Paged Attention Provider；默认 reference
   仍有完整 layer K/V copy 和随上下文增长的 workspace，D1 性能未验收。
2. online backend 的真实 8B 多页验证有 24/297 项失败且误差继续传播；必须定位
   舍入/累计契约或由 fused backend 建立独立 contract，不能放宽门禁后默认启用。
3. materialized SDPA 的 Transformer 阶段已一致，但真实 8B streamed LM Head 仍有
   2/297 项 logits/Top-k 门禁失败，需要单独形成 golden regression。
4. 4K/8K/32K/64K/128K 的真实长上下文正确性、显存和性能尚未执行。
5. 当前 attention wall time 是同步 wall-clock microbenchmark；RunReport 中还没有
   runtime-owned CUDA Event 级 KV kernel 分段。

### 后续里程碑

- D2 Prefix Hash、Salt、最长匹配和 LRU 未实现。
- D3 TransferArbiter、Pinned CPU page pool 和 restore/recompute model 未实现。
- D4 INT8 KV 仅完成内存预算，没有页面格式、scale 或 kernel。
- D5 Quest Flat 未实现，当前 `kv-index=quest-flat` 会明确拒绝。
- D6 NVMe Persistent Prefix 未实现，不存在逐 Token SSD read。
- D7 Hierarchical Quest/Delta Index 未实现。
- D8 Active KV Offload 仍为实验研究，不承诺性能。

在 P0 的 fused dense backend、online 数值合同和真实 8B LM Head Golden 关闭前，
不应把 D1 标记为
`qualified`，也不应提前将 sparse 或 NVMe 路径设为默认。
