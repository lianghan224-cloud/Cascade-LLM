# KV Framework V1 RC 旧版资格报告（Numerical Contract V1）

> 本文仅保留历史审计。当前结论以
> `KV_FRAMEWORK_V1_PRODUCTION_QUALIFICATION.md` 的 Numerical Contract V2
> 复验为准。

**资格日期：** 2026-08-02

**候选版本：** KV Framework V1 RC

**测试分支：** `framework-v0`

**主要实机：** 2 × NVIDIA GeForce RTX 3080 Ti 12GB（SM86）

**软件：** PyTorch 2.4.1+cu121、CUDA Runtime 12.1

## 1. 最终结论

本轮结论是：

```text
KV Framework V1 RC:                         通过
KV Framework V1 Production Qualified:       未通过
```

V1 的 Page、Request、Batch、Store、Selection、Reuse、Attention Backend、
Page Kernel Backend 和 Provider Bundle 边界已经冻结并受回归测试保护；SM86 的直接
Paged CUDA 路径可以完成真实 Llama-3.1-8B 运行，且没有隐藏的完整 KV gather、SDPA
fallback 或随上下文线性增长的 workspace。

不能升级为 Production Qualified 的两个硬门禁是：

1. 真实模型相对 Hugging Face SDPA 的严格逐阶段 elementwise contract 尚未全部通过；
   误差已经归因，但“解释”不等于“通过”。
2. 当前机器没有 SM80、SM89 或 SM90，三者只能证明独立 Bundle/Capability/加载路径，
   不能证明真机执行、数值或性能资格。

没有放宽既有 `atol=0.05`、`rtol=0.05`，没有更新 Golden，也没有把 Top-1 一致替代
严格阶段门禁。

## 2. 生产资格门禁

| 门禁 | 结果 | 证据/说明 |
|---|---|---|
| 自动化测试全部通过 | 通过 | 137 tests；CUDA 隔离运行中 13 项按环境跳过，完整 CUDA 回归见第 9 节 |
| Provider ABI 边界正确 | 通过 | Attention 与 append/copy 已拆分，生命周期只属于 Runtime/PagePool |
| Production 无隐藏 gather/SDPA | 通过 | 源码门禁 + production workspace 为 0 |
| Kernel Numerical Contract | 通过 | SM86 MHA/GQA/MQA、prefill/decode 六类均通过 |
| 真实 8B Greedy | 通过 | 短序列三个位置 Top-1 一致 |
| 真实模型严格阶段输出 | **未通过** | 8B 为 23/297 阶段失败；TinyLlama 为 11/207 |
| 真实 8B 1000-token 长稳 | 通过 | 真实自回归、页面/显存/数值及 HF 抽样回放均通过 |
| Fork/COW/Prefix/Beam/Speculative 长稳 | 通过 | 1000-cycle 精确 ownership graph 校验 |
| 代表性场景快于 Reference | 通过 | 短/中生产路径最低为 Reference 的 53.50×；长场景单独与 Generic 对照 |
| SM86 真机资格 | `smoke_passed` | 真机 kernel 与真实模型可运行；严格模型 Golden 仍阻止 qualified |
| SM80/SM89/SM90 真机资格 | `unqualified` | 当前没有对应物理 GPU，未执行跨架构 kernel |

## 3. Provider 边界审计与修复

审计前，页面生命周期已经位于 `PagedKVRuntime` / `KVPagePoolV1`，Provider 没有接管
allocate、release、fork、COW、ref_count 或 pin_count。因此没有推倒 D0/D1。

需要修复的是旧 Attention Provider 同时暴露 attention 与 page payload 操作。冻结后的
结构为：

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

边界门禁包括：

- Attention Backend 暴露 append/copy 时拒绝注册；
- Page Kernel Backend 暴露 attention 方法时拒绝注册；
- 任一 Backend 暴露生命周期方法时拒绝注册；
- 不创建 Provider 也能单测 PagePool allocate/retain/release；
- 替换 Attention Backend 不改变 Block Table、PageHandle 或引用计数；
- 替换 Page Kernel Backend 后 partial-tail COW 语义不变。

兼容导入别名仍保留，但旧 `PagedAttentionProvider` 别名现在只指向 attention-only ABI。
这是 RC 阶段的边界纠正：KV contract fixture SHA 从预冻结值更新为
`c1b8e8cad65c0d0d1779703e24129a832100efbd5024a6cfad38841f8e30c0ed`。项目尚未发布
Production V1，因此本轮选择在生产冻结前修正合同；ExecutionPlan schema v1 和
RunReport schema v2 均未改义。此后再次改变这些边界必须提升对应 ABI。

## 4. 严格阶段失败的完整解释

原始逐阶段列表位于：

```text
real_results/kv_v1/real8b_numerical_attribution.json
real_results/kv_v1/tinyllama_numerical_attribution.json
```

### 4.1 8B

- 23 个 production 阶段失败全部可以追溯到每个 phase 的 layer 0 attention 首次非零差异；
- first difference 的 max/mean absolute error：
  - prefill：`0.0009765625 / 6.04418e-6`；
  - decode-0：`0.000244140625 / 4.97979e-6`；
  - decode-1：`0.0001220703125 / 7.58522e-6`；
- 失败分布：hidden 13、MLP 4、final norm 3、logits 3；
- 19/23 相同阶段在官方 HF SDPA 与 HF eager 的控制实验中也超过同一严格门限；
- 余下 4 项仍位于同一 phase 的 layer-0 attention 差异之后；
- 3 个 logits 位置 Top-1 全部一致。

### 4.2 TinyLlama

- 11 个 production 阶段失败全部有 layer-0 attention 上游差异；
- HF SDPA 与 eager 的相同阶段也全部失败；
- 失败分布：hidden 3、MLP 2、final norm 3、logits 3；
- 3 个 logits 位置 Top-1 全部一致。

### 4.3 原因与边界

Production CUDA kernel 直接读取非连续 HND 页面，以 FP32 两遍 reduction 执行 QK、
softmax 和 V 累积；HF SDPA 使用私有 fused reduction tree。最初约 `1e-4～1e-3`
的 attention 输出差异经过残差、SwiGLU、RMSNorm 和 LM Head 放大。BF16 checkpoint
没有量化 scale，因此此前 fused/fallback 排查表中的“weight scale 范围”在本项为
不适用；矩阵 placement 也不是原因，因为：

- resident 与 streamed production 路径 297/297 bitwise 一致；
- `legacy_gather_sdpa_reference` 使用相同 HF SDPA 后，streamed/resident 均 297/297 通过；
- 独立 HF SDPA/eager 控制实验本身也产生深层阶段差异。

这排除了 Page mapping、COW、resident/streamed layout 和 LM Head placement，但没有让
严格 HF-SDPA identity 变成通过。若未来接受 architecture/provider-specific 模型
contract，必须独立评审并版本化，不能在本报告中临时放宽旧 contract。

## 5. 真实 8B 1000-token 长稳

配置为真实 Llama-3.1-8B-Instruct BF16 checkpoint、SM86 direct-paged provider、
streamed Transformer/Embedding/LM Head、page size 16，从 BOS token `128000` 开始执行
真实 Greedy 自回归，而不是喂入合成 token IDs。

```text
generated tokens:                 1000
duration:                      1464.34 s
mean / p50:              1459.61 / 1458.87 ms/token
first quartile mean:            1459.88 ms/token
last quartile mean:             1459.06 ms/token
last / first quartile ratio:      0.99944
CUDA allocated drift:                   0 bytes
CUDA reserved drift:                    0 bytes
thread drift:                            0
final logical pages:                    63
final total ref_count:                  63
quiescent pin_count:                     0
CUDA Event count:                       64
workspace peak:                          0 bytes
provider fallback:                    null
```

所有 101 个采样位置的 hidden state 和 Top-k 均为有限值，hidden RMS 位于
`0.5951～2.8757`。请求释放后 63 页全部返回池；随后 100 次额外 request 页面分配/释放
也没有残留。

HF 使用同一初始 token 和 Cascade 生成的同一 1000-token 路径进行逐 token replay：

```text
HF replayed tokens:                    1000
sampled positions:                      101
Top-1 agreement:                    101/101
minimum Top-10 set consistency:          0.8
maximum sampled Top-k value error:       1.0
ordered Top-10 exact matches:          44/101
HF logits finite:                       true
```

这证明长运行中没有 Greedy 漂移，但 ordered Top-k 和 logits 数值仍存在与第 4 节一致的
attention reduction 差异，不能拿 101/101 Top-1 覆盖严格阶段 Golden 的失败。

原始数据：

```text
real_results/kv_v1/real8b_sm86_1000_token_soak.json
real_results/kv_v1/real8b_sm86_1000_token_hf_replay.json
```

## 6. Prefill/Decode 性能矩阵

短/中矩阵采用 batch 2、GQA 32 query heads / 8 KV heads、head dim 128、prefill
query length 8、page size 16/32、warmup 2、正式 5 次；context 为 16/128/512。

```text
case count:                              36
all supported:                         true
production Numerical Contract:        true
maximum production workspace:       0 bytes
production vs Reference:       53.50×～2172.26×
SM86 vs Generic:                 1.01×～1.75×
```

代表性 SM86 P50：

| Context | Page | Decode P50 | 8-token suffix Prefill P50 |
|---:|---:|---:|---:|
| 16 | 16 | 0.484 ms | 0.466 ms |
| 128 | 16 | 0.662 ms | 0.740 ms |
| 512 | 16 | 1.390 ms | 1.713 ms |
| 512 | 32 | 1.329 ms | 1.654 ms |

长矩阵采用 batch 1、prefill query length 4、warmup 1、正式 3 次；不运行 Python
Reference，避免把解释器循环当成长上下文性能基线。

| Context | Page | SM86 Decode P50 | SM86 Prefill P50 | SM86 / Generic 范围 |
|---:|---:|---:|---:|---:|
| 4K | 16/32 | 9.07～9.36 ms | 8.12～9.38 ms | 1.13×～1.20× |
| 8K | 16/32 | 15.77～16.37 ms | 15.90～16.48 ms | 1.12×～1.19× |
| 32K | 16/32 | 62.12～64.41 ms | 62.69～64.95 ms | 1.13×～1.20× |

长矩阵 24/24 case 支持，全部 workspace 为 0。它只测 Paged Attention kernel，不能
代表完整 8B streamed-weight token latency，也不能与 FlashInfer/vLLM 直接排名；完整
8B decode 仍由每 token 权重流式传输主导，约 1459.6 ms/token。

原始数据：

```text
real_results/kv_v1/production_matrix_short_medium.json
real_results/kv_v1/production_matrix_long.json
```

## 7. Fork/COW/Prefix/Beam/Speculative 长稳

`tools/soak_kv_lifecycle_v1.py` 在真实 CUDA PagePool 上完成 1000 个循环：

```text
Beam fork/discard:             1000
Speculative fork/commit:       1000
Session fork:                   142
Prefix register/lookup:          62
ownership graph mismatch:         0
quiescent pin count:              0
CUDA allocated drift:             0 bytes
CUDA reserved drift:        2097152 bytes
thread drift:                      0
close 后 allocated pages:          0
```

每次采样都从所有活跃 Request、Session 与 Prefix owner 重建期望 ownership graph，并逐页
比较 generation-safe handle 和 descriptor.ref_count；不是只比较最后的空闲页数量。

原始数据：`real_results/kv_v1/lifecycle_fork_cow_prefix_1000.json`。

## 8. 硬件资格

| 架构 | 物理硬件 | Bundle 加载 | 真机执行 | 数值 | 状态 |
|---|---|---|---|---|---|
| SM80 | 无 | 通过 | 未执行 | 未执行 | `unqualified` |
| SM86 | RTX 3080 Ti | 通过 | 通过 | 六类 kernel contract 通过 | `smoke_passed` |
| SM89 | 无 | 通过 | 未执行 | 未执行 | `unqualified` |
| SM90 | 无 | 通过 | 未执行 | 未执行 | `unqualified` |

SM86 覆盖 MHA/GQA/MQA × prefill/decode，最大 absolute error 为 `0.015625`，所有 case
低于已冻结的 architecture/provider contract，workspace 为 0，fallback reason 为 null。

SM80/89/90 不能通过在 SM86 上加载 Python class 来取得资格。对应 Bundle 能加载和在
启动前拒绝错误架构，只证明框架路径存在；真机、驱动、PTX/SASS、数值和性能全部未验证。

原始数据：`real_results/kv_v1/hardware_provider_qualification_sm86_host.json`。

## 9. 自动化与复现

完整 CUDA 环境回归结果：

```text
Ran 137 tests in 22.041s
OK
```

另一次 `CUDA_VISIBLE_DEVICES=''` 隔离运行同样 137/137 通过，13 项 CUDA-only 测试按
环境跳过；这证明 lifecycle/contract 测试不依赖 Provider 或 GPU。

核心复现入口：

```bash
PYTHONPATH=tests .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

.venv/bin/python tools/soak_kv_v1.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B-Instruct \
  --decode-tokens 1000 --provider sm86 --token-source generated \
  --initial-token-id 128000 \
  --output real_results/kv_v1/real8b_sm86_1000_token_soak.json

.venv/bin/python tools/soak_kv_lifecycle_v1.py \
  --cycles 1000 \
  --output real_results/kv_v1/lifecycle_fork_cow_prefix_1000.json

.venv/bin/python tools/qualify_paged_hardware.py \
  --output real_results/kv_v1/hardware_provider_qualification_sm86_host.json

.venv/bin/python tools/benchmark_kv_v1.py \
  --providers reference_paged_exact,generic_cuda,sm86 \
  --lengths 16,128,512 --page-sizes 16,32 \
  --phases decode,prefill --batch-size 2 --prefill-query-length 8 \
  --warmup 2 --runs 5 \
  --output real_results/kv_v1/production_matrix_short_medium.json
```

## 10. 未完善或有缺陷的部分

1. 严格 HF-SDPA 阶段 identity 未通过，是当前生产资格的 P0 数值缺口。
2. SM80/89/90 缺少对应真机，保持 `unqualified`。
3. SM86 kernel 是 correctness-first、非 tiled 的两遍实现；没有达到 FlashInfer/vLLM
   等成熟 Paged Attention kernel 的性能水平。
4. Executor 仍是单请求适配层；Runtime ABI 支持 ragged batch，但 continuous batching
   scheduler 未实现。
5. Prefix Cache 尚无容量上限、LRU、tenant-facing API 和 partial-block reuse。
6. INT8/FP8/INT4 KV、CPU/NVMe 活跃 KV、Quest 和 Hierarchical Quest 仍是明确 unsupported。
7. CUDA Graph 未实现；Batch metadata 构造仍有 host synchronization。
8. SM86 专用 Provider 当前复用同一正确性 kernel 及不同 launch policy，不是成熟的架构
   专用 tiled kernel。

因此本版本应发布为 **KV Framework V1 RC**。只有严格模型 contract 通过、SM86 完整
资格闭环，且目标发布架构分别取得真机证据后，才可更新为 Production Qualified。
