# KV Framework V1 验证与消融报告

**日期：** 2026-08-02
**硬件：** 2 × RTX 3080 Ti 12GB，测试设备 SM86
**软件：** PyTorch 2.4.1+cu121，CUDA Runtime 12.1，Python 3.8.10

## 1. 结论

Page/Request/Batch/Store/Reuse V1 骨架、分离的 Attention/Page Kernel Backend、Generic
CUDA、SM86、共享/COW、确定性 LM Head、内存预算和 RunReport 已接通。自动化与 kernel
数值/性能门禁通过；真实 Llama-3.1-8B 和 TinyLlama 的所有短序列位置 Greedy Top-1
一致，8B 1000-token 同路径 HF 抽样也达到 101/101 Top-1 一致。

本轮不能声明完整 production qualification：真实模型相对 HF fused SDPA 的严格
elementwise/ordered Top-k Golden 仍未全部通过，SM80/89/90 也没有真实硬件验证。

## 2. 自动化回归

```text
137 / 137 tests passed
```

新增覆盖：

- generation/stale handle、页面状态和可控容量错误；
- 随机非连续物理页、partial tail、page 16/32；
- ragged Batch、不同 sequence/query length；
- MHA/GQA/MQA prefill/decode；
- Fork、partial-tail COW、Beam、speculative commit/discard；
- Session 和 namespace-isolated in-memory Prefix；
- 显式 Reuse Policy 分发，request-only 不会静默获得跨请求复用；
- Selection 的 logical block/valid-token 元数据及非连续“首+尾页”CUDA 验证；
- GPU Store `read_pages/write_pages` round-trip；
- BF16/FP16 Generic CUDA；
- SM86 真实 kernel 和 SM80/89/90 独立 capability 状态；
- resident/streamed LM Head bitwise 一致；
- V1 合同 fixture/hash。
- Attention Backend 与 append/copy Page Kernel Backend 边界；
- 无 Provider PagePool lifecycle 与 Backend 热替换 COW 语义；
- production source 禁止隐藏 gather/SDPA fallback。

## 3. Kernel 数值与性能消融

本轮干净复测数据：

```text
real_results/kv_v1/production_matrix_short_medium.json
real_results/kv_v1/production_matrix_long.json
```

完整矩阵包含：context 16/128/512、page 16/32、decode/prefill、batch=2、
32 query heads、8 KV heads、head_dim 128。共 36 个 provider case。

结果：

- Generic 和 SM86 全部通过 architecture/provider/dtype keyed Numerical Contract；
- 所有 production case `workspace_peak_bytes = 0`；
- 所有 production case 快于 Python `reference_paged_exact`；
- 本轮 SM86 相对 Generic 为 `1.01×～1.75×`；
- 最大 kernel pairwise error 不超过 `0.001953125`；
- 4K 的 Generic/SM86 decode 与 8-token suffix prefill 仍保持 0 workspace；
- 4K/8K/32K 长矩阵的 24 个 production case 全部 0 workspace；
- 长矩阵 SM86 相对 Generic 为 `1.12×～1.20×`。

Reference 是 Python correctness backend，因此数十到上千倍的 speedup 不能与
FlashInfer、vLLM 或 FlashAttention 横向比较。有效的本轮结论只有 production 比
当前 production providers 均快于 Reference；SM86 调优配置在多数 shape 上比同源码
Generic 快。两者仍是 correctness-first kernel，不能据此声明行业领先。

复现：

```bash
.venv/bin/python tools/benchmark_kv_v1.py \
  --lengths 16,128,512 \
  --page-sizes 16,32 \
  --phases decode,prefill \
  --batch-size 2 \
  --prefill-query-length 8 \
  --warmup 2 --runs 5 \
  --output real_results/kv_v1/production_matrix_short_medium.json
```

## 4. Streamed LM Head 消融

LM Head 使用固定 FP32 reduction tree，并按 BF16/FP16 `linear` 语义写回 native dtype 后：

- BF16/FP16 不同 chunk 划分 bitwise 一致；
- BF16 synthetic golden 与显式 FP32 dot + BF16 rounding bitwise 一致；
- FP16 固定 golden 最大误差为 `2^-18`（不同 FP32 reduction order）；
- tied/untied tiny checkpoint 的 resident 与 streamed logits bitwise 一致；
- 真实 8B 的 resident/streamed 消融产生完全相同的误差统计。

使用 `legacy_gather_sdpa_reference` 恢复与 HF 相同的 Transformer SDPA 路径后，
297 个阶段全部通过原有 HF elementwise/ordered Top-k 门禁：

| Stage | max abs | mean abs | Top-1 |
|---|---:|---:|---|
| prefill logits | 0.0625 | 9.79865e-6 | equal |
| decode-0 logits | 0.015625 | 3.28848e-7 | equal |
| decode-1 logits | 0.015625 | 3.57911e-7 | equal |

streamed 与 resident 的 canonical `.comparisons` SHA-256 均为
`1fd052087da01dbbe4e9ce64d01d2cc2f791e851c565c161364da4774e7e7baa`。
修复只补上 checkpoint compute dtype 要求的最终舍入；原有 `atol=0.05`、`rtol=0.05`
和 Golden 均未修改。

原始数据：

```text
real_results/kv_v1/real8b_legacy_gather_lm_head_ablation.json
real_results/kv_v1/real8b_legacy_gather_resident_lm_head_ablation.json
```

## 5. 真实模型

### Llama-3.1-8B-Instruct

配置：17-token 跨页 prefill，2-step decode，SM86 provider，BF16 streamed weights 和
streamed vocab。297 个阶段中 23 个未通过旧 HF elementwise/ordered Top-k 门禁；三个
生成位置 Top-1 全部一致，Top-10 set consistency 最低 0.9。

两遍 Paged FP32 与 HF fused SDPA 的不同归约会在 32 层后传播。全部 23 项失败已经
逐项归因：每个 phase 的首次差异均为 layer-0 attention；HF SDPA/eager 控制实验本身
也有 20 个严格失败，其中 19 个与 production 失败同阶段。因此该结果必须标记为：

```text
kernel contract: passed
greedy token: passed
strict HF model golden: failed
```

原始数据：`real_results/kv_v1/real8b_sm86_multipage.json`。
归因数据：`real_results/kv_v1/real8b_numerical_attribution.json`。

### TinyLlama-1.1B-Chat-v1.0

真实单文件 safetensors，22 层、32 query heads、4 KV heads、head_dim 64。Checkpoint
在 GPU 分配前完整校验通过；15-token prefill + 2 decode 的 207 个阶段中 11 个未通过
旧 HF elementwise/ordered Top-k 门禁，但所有位置 Top-1 一致，Top-10 set consistency
最低 0.9。

真实生成入口以 `Hello` 完成 2-token prompt + 4 token 生成：

```text
TTFT:                    606.12 ms
decode:                  201.31～217.62 ms/token
GPU peak allocated:      273,409,536 bytes
KV pool:                 360,448 bytes
KV workspace:            0 bytes
logical appended tokens: 5
layer-token writes:      110
CUDA Event count:        44（22 layer × 2）
paged provider:          generic_cuda
LM Head backend:         deterministic_cuda_fp32_accum_native_output
```

原始数据：

```text
real_results/kv_v1/tinyllama_real_generic_cuda.json
real_results/kv_v1/tinyllama_run_report.json
real_results/kv_v1/tinyllama_numerical_attribution.json
```

## 6. 长稳和生命周期

原有 synthetic 数据：

```text
real_results/kv_v1/soak_1000_tokens.json
real_results/kv_v1/soak_100_load_cycles.json
```

2-layer MQA synthetic Llama 完成连续 1000-token decode 和 100 次额外页面创建/释放：

```text
CUDA allocated drift: 0
CUDA reserved drift:  0
thread drift:         0
CUDA Event count:     4 → 4（2 layer，各 1 个 Append + Attention Event）
workspace peak:       0
all pages released:   true
```

独立 100 次模型/store/runtime/KV load-close 循环：

```text
CUDA allocated drift: 512 bytes（低于 8 MiB 门限）
CUDA reserved drift:  0
thread drift:         0
CUDA Event count:     4 → 4
all pages released:   true
```

真实 Llama-3.1-8B SM86 路径完成 1000-token 自回归：

```text
mean / p50:             1459.61 / 1458.87 ms/token
last/first quartile:    0.99944
CUDA allocated drift:   0
CUDA reserved drift:    0
thread drift:           0
final page/ref:          63 / 63
quiescent pins:         0
workspace:              0
all pages released:     true
HF sampled Top-1:       101 / 101
```

独立生命周期 soak 还完成 1000 次 Beam fork/discard、1000 次 speculative fork/commit、
142 次 Session fork 和 62 次 Prefix register/lookup；每次采样的 ownership graph 与
descriptor.ref_count 精确一致，关闭后页面归零。

原始数据：

```text
real_results/kv_v1/real8b_sm86_1000_token_soak.json
real_results/kv_v1/real8b_sm86_1000_token_hf_replay.json
real_results/kv_v1/lifecycle_fork_cow_prefix_1000.json
```

## 7. 验收矩阵

| 项目 | 状态 | 说明 |
|---|---|---|
| V1 Page/Request/Batch/Backend/Bundle ABI | 通过 | fixture + SHA-256 冻结 |
| 生命周期与 Provider 边界 | 通过 | Runtime 独占 lifecycle；Attention/Page Kernel 已拆分 |
| 非连续 HND page direct attention | 通过 | Reference/Generic/SM86 |
| 无完整 K/V 和 score workspace | 通过 | production 0 bytes |
| MHA/GQA/MQA、prefill/decode、ragged batch | 通过 | synthetic contract |
| Fork/COW/Beam/speculative KV primitives | 通过 | 集成测试 |
| Session/in-memory Prefix foundation | 通过 | full sealed blocks |
| Generic CUDA | 通过 | SM86 实机；其他架构待资格 |
| SM86 性能优势 | 通过（本轮矩阵） | 对 Generic 1.01×～1.75×；尚非成熟 tiled kernel |
| Llama-3.1-8B Greedy | 通过 | 三个位置 Top-1 一致 |
| 第二真实 GQA 模型 Greedy | 通过 | 三个位置 Top-1 一致 |
| LM Head legacy-HF 严格门禁 | 通过 | streamed/resident 均 297/297 |
| Production Paged 严格 HF 门禁 | **未通过** | 不放宽门限；kernel contract 另行通过 |
| 真实 8B 1000-token 稳定性 | 通过 | 0 resource drift，HF sampled Top-1 101/101 |
| SM80/SM89/SM90 真实资格 | **未完成** | `declared` |
| INT8/FP8/INT4 KV、Offload、Quest | **未实现** | ABI only |
| Continuous batching scheduler | **未实现** | Runtime ABI 已支持 Batch |

## 8. 下一步

1. 评审并建立真实 8B architecture/provider-specific model contract，或实现与 HF SDPA
   一致的 reduction；不能用 synthetic contract 或 Top-1 一致替代严格阶段门禁。
2. 优化 tiled prefill/decode 并与 FlashInfer/vLLM 在相同 shape 上比较。
3. 在真实 SM80/SM89/SM90 上运行相同 suite 后再升级 compatibility 状态。
4. 接入 continuous batching scheduler；不修改 V1 Page/Batch/Backend ABI。
