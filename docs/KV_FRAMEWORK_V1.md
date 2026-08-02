# Cascade-LLM KV Framework V1

KV Framework V1 将页面、请求、Batch、Store、Selection、Reuse、Paged
Attention Backend、Paged KV Kernel Backend 和 Provider Bundle 的边界冻结为版本化合同。它取代 D0/D1 中以
`layer_streaming/kv_cache.py` 为主的旧 reference 架构；旧模块只保留兼容和消融。

## 1. 当前可执行范围

当前真实执行主路径是：

```text
GPU Store
+ BF16/FP16 HND pages
+ exact dense selection
+ request/session/in-memory sealed-prefix sharing
+ paged prefill/decode
+ MHA/GQA/MQA
+ ragged Batch metadata
+ generic CUDA or SM86 provider
```

以下只冻结 ABI，不实现数据搬运或 kernel：

- pinned CPU/NVMe active KV Store；
- INT8/FP8/INT4 KV；
- Quest Flat、Hierarchical Quest；
- persistent prefix、实时 SSD decode；
- CPU/GPU split attention。

选择未实现组合会在 Page Pool 分配前失败。Production Provider 不会自动调用
`legacy_gather_sdpa_reference`，Reference 必须同时显式选择并传入
`--allow-kv-reference`。

## 2. 冻结合同

机器可读合同位于：

```text
tests/fixtures/kv_framework_v1.json
```

合同 SHA-256：

```text
c1b8e8cad65c0d0d1779703e24129a832100efbd5024a6cfad38841f8e30c0ed
```

冻结项包括：

1. generation-safe `PageHandle`；
2. `PageDescriptor` 和页面状态机；
3. 每层 `[page, kv_head, page_token, head_dim]` HND 布局；
4. `RequestKVState` 和 storage-independent `LogicalBlockTable`；
5. `PagedBatchView`、flat block table、逻辑块位置、页内有效长度和 `SlotMapping`；
6. Store `read_pages/write_pages/copy_page`、Selection、Reuse
   `fork/register_prefix/lookup_prefix` ABI；
7. Batch-first `PagedAttentionInput/Output` 和 `PagedAttentionBackend` ABI；
8. 只含 `append_kv/copy_pages` 的 `PagedKVKernelBackend` ABI；
9. 聚合两者但不拥有生命周期的 `PagedProviderBundle`；
10. Provider Capability 和 architecture-specific Numerical Contract。

不兼容变更必须升级 ABI/format version，不能只更新 fixture。

## 3. 事务与共享语义

一次 token append 是跨全部 Transformer layer 的事务：

```text
begin append
  → 分配缺失页面 / partial tail COW
  → 每层写入相同 Slot Mapping
  → 全部 layer 完成后 commit + seal
  → 任一 layer 失败则 rollback
```

请求 Fork 共享 sealed 页面并增加 `ref_count`。共享 partial tail 在下一次写入前
复制全部 layer 的有效 K/V，父子请求随后独立。Beam 和 speculative branch 使用同一
`fork / commit_branch / discard_branch` 原语。页面被 CUDA Attention 使用时增加
`pin_count`，固定 per-layer Append/Attention Event 完成后解除 pin；被 pin 或处于 copy/migrate 状态的
页面不能释放。

内存 Prefix Cache 只注册完整 sealed token block，使用 namespace、父 block hash、
逻辑位置和 token IDs 的链式 SHA-256。Namespace 应由调用方包含 model、tokenizer、
RoPE、权重量化、KV format/layout、adapter/LoRA 和 tenant salt。

## 4. Provider Bundle 与边界

冻结后的硬件 Bundle 结构是：

```text
KV Runtime / Page Pool
  allocate / release / fork / COW / ref_count / pin_count

PagedProviderBundle
├── PagedAttentionBackend
│   ├── prefill
│   ├── decode
│   └── estimate_workspace
└── PagedKVKernelBackend
    ├── append_kv
    └── copy_pages
```

Attention/Kernel Backend 均禁止暴露 allocate、release、fork、retain、pin 或 unpin。
替换任一 Backend 后，Block Table、PageHandle generation 和引用计数语义保持不变；
这些边界由无 Provider PagePool 测试和 Backend 热替换 COW 测试门禁。

## 5. Paged Attention Backend

### `reference_paged_exact`

两遍 FP32 page-wise softmax：第一遍求全局 max，第二遍求 denominator 和 V 累积。
它直接遍历非连续 Block Table，不拼接完整 K/V，也不创建完整 score matrix；由于包含
Python page/head 循环，只用于正确性。

### `generic_cuda`

使用 NVRTC 编译、CUDA Driver API 加载，无需 nvcc、C++ compiler 或 Python headers。
CUDA kernel 直接读取 HND page 和 flat block table，使用两遍 FP32 softmax；全局
workspace 为 0。支持 SM80/86/89/90、BF16/FP16、page size 16/32、MHA/GQA/MQA、
ragged prefill/decode 和 partial tail。

Attention Backend 消费 Selection 输出的物理页 ID、原逻辑块 ID 和页内有效 token 数；因此
Quest 将来可以返回非连续历史页而不修改 Batch/Backend ABI。当前用“第一页 + 尾页”的
非连续选择做了 Generic CUDA 接口验证，但这不代表 Quest 评分算法已经实现。

### Architecture Provider Bundle

| Provider | 真实硬件状态 | 实现 |
|---|---|---|
| `sm80` | `unqualified` | 独立 capability/加载路径，复用 Generic kernel |
| `sm86` | `smoke_passed` | head_dim 128 专用 128-thread reduction |
| `sm89` | `unqualified` | 独立 capability/加载路径，复用 Generic kernel |
| `sm90` | `unqualified` | 独立 capability/加载路径，复用 Generic kernel |

`sm80/sm89/sm90` 不能描述为已兼容；它们必须在对应真实硬件上完成模型、长稳和性能
资格验证后才能提升状态。

## 6. LM Head 数值路径

Streamed 和 resident LM Head 现在都使用
`deterministic_cuda_fp32_accum_native_output`。每个 vocab row 使用固定 256-thread FP32 reduction
tree，因此结果不随 vocab chunk 行数或 resident/streamed placement 改变。该路径避免
了 full-vocab 与 chunked cuBLAS 选择不同 GEMM 算法造成的 Golden 抖动。
累积完成后按权重的 BF16/FP16 dtype 舍入，再由报告路径按需转换为 FP32；这与
PyTorch `linear` 的输出 dtype 语义一致。

它当前优先数值可复现性，尚未证明比 cuBLAS 更快；LM Head 性能必须独立报告，不能把
这一正确性修复描述为性能提升。

## 7. CLI

Production：

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint MODEL \
  --kv-accuracy exact \
  --kv-storage gpu \
  --kv-dtype bf16 \
  --kv-index none \
  --kv-prefix-cache off \
  --kv-page-size 16 \
  --kv-attention-backend generic_cuda
```

SM86、Llama head_dim 128：

```bash
--kv-attention-backend sm86
```

Reference 诊断：

```bash
--kv-attention-backend reference_paged_exact --allow-kv-reference
```

## 8. 扩展规则

后续能力只能按以下方式增加：

```text
KV 量化        → Format + Attention/KV Kernel Backend
CPU/NVMe       → Store
Prefix 持久化  → Store + Reuse Policy
Quest          → Selection Policy
Split Attention → Attention Backend
```

不得再建立第二套 Page Pool、Block Table、Request State 或单请求 Kernel ABI。

## 9. 当前缺陷

1. Llama Executor 仍是单请求 adapter；KV Runtime/Provider 已支持 ragged Batch，完整
   continuous batching scheduler 尚未接入。
2. Generic/SM86 kernel 是 correctness-first 两遍实现，prefill 没有 tile/GEMM 化，
   超长上下文性能距离 FlashAttention/PagedAttention 生产实现仍有差距。
3. 当前 architecture-specific Numerical Contract 只门禁单次 kernel；HF fused SDPA
   使用不同归约路径，真实模型深层 elementwise 误差仍单独标红，不能用 kernel contract
   替代模型 Golden。
4. Prefix Index 尚无容量上限、LRU、tenant API 和 partial-block reuse。
5. CUDA Graph 未实现；Batch metadata 当前会在 host 构造并有少量同步点。
6. SM86 已完成 RTX 3080 Ti kernel smoke、性能矩阵和真实 8B 1000-token 长稳，但严格
   HF-SDPA 阶段 Golden 仍失败，因此状态仍不是 `qualified`。
7. SM80/SM89/SM90 没有真实硬件结果，必须保持 `unqualified`。
8. Quantized KV、CPU/NVMe Store 和 sparse selection 都是明确的 unsupported 接口。

验证数据和未通过项见 `docs/KV_FRAMEWORK_V1_VALIDATION.md`。
