# Llama-3.1-8B 矩阵级双缓冲实现报告

测试/实现日期：2026-07-25。

## 完成范围

本轮在原有校准基准之外新增了可复用的 `layer_streaming` 运行时：

- `full_pinned`：完整 checkpoint 保存在 BF16 page-locked CPU arena；
- `pinned_staging`：完整 checkpoint 保存在 pageable CPU arena，两个 CPU
  worker 将后续权重复制到两个可复用 pinned staging slot；
- 默认 `matrix` 粒度：Q/K/V/O/Gate/Up/Down 分别作为传输单元；
- 可回退 `layer` 粒度：一整个 Transformer 层作为传输单元；
- GPU slot 大小由当前模型计划中的最大传输单元自动决定；
- 保留普通两个 device slot、一个 H2D stream、一个 compute stream；
- Embedding、LM Head、每层 RMSNorm 和最终 RMSNorm 使用独立 GPU
  resident arena，不进入循环 H2D；
- 支持 Llama-3.1-8B 本地 safetensors 单文件和 sharded index；
- 提供单请求 Prefill 及逐 Token Decode 执行器和简单 GPU KV Cache。

## Llama-3.1-8B 内存计划

默认配置为独立的 Embedding 和 LM Head，即
`tie_word_embeddings=false`。

| 项目 | 整层基线 | 默认矩阵粒度 |
|---|---:|---:|
| 传输单元数量 | 32 | 224 |
| 最大传输单元 | 416 MiB | 112 MiB |
| 两个 GPU weight slot | 832 MiB | 224 MiB |
| GPU resident权重 | 2004.508 MiB | 2004.508 MiB |
| 权重相关GPU分配合计 | 2836.508 MiB | 2228.508 MiB |

结果：

- 绝对节省：608 MiB；
- 流式权重缓冲减少：73.08%；
- 包含LM Head/Embedding常驻区后，权重相关GPU显存减少：21.43%；
- 按BF16 KV Cache约128 KiB/Token计算，608 MiB约可容纳额外4864个Token。

如果checkpoint明确绑定Embedding和LM Head，计划会让两者共享resident
view，可再减少1002 MiB；官方Llama-3.1-8B默认按不绑定处理。

## 两种CPU内存模式

| 模式 | Pageable权重 | Pinned CPU内存 | 特点 |
|---|---:|---:|---|
| `full_pinned` | 0 | 约14.96 GiB | H2D直接读取，默认高性能模式 |
| `pinned_staging` | 约14.96 GiB | 224 MiB | 节省锁页内存，需要CPU staging吞吐 |

`pinned_staging`总RAM会比权重payload多两个112 MiB staging slot。两个
worker与两个slot一一流水复用；host slot在对应H2D Event完成后即可覆写，
不必等待GPU计算完成。

本机此前单线程pageable→pinned memcpy约14.8 GB/s，而维持现有H2D
流水需要约23.6 GB/s聚合staging吞吐。因此：

- 单CPU worker一定会成为瓶颈；
- 默认使用两个worker；
- 两worker是否能稳定超过23.6 GB/s必须在GPU恢复可见后实测；
- `pinned_staging`是兼容模式，不能预先宣称与`full_pinned`等速。

## 性能对比

已有Llama-3.1-8B、BF16、M=1、精确Projection形状基线：

| 项目 | 时间 |
|---|---:|
| 整层串行 | 610.312 ms |
| 整层双缓冲 | 592.108 ms |
| 整层双缓冲相对串行 | 1.0307× |

矩阵粒度不改变Decoder每Token约13.0 GiB权重payload，只把copy提交次数从
32增加到224。利用已有提交成本估算：

| 口径 | 预测矩阵双缓冲 | 相对串行 | 相对整层双缓冲 |
|---|---:|---:|---:|
| 仅计Raw Driver额外提交 | 592.461 ms | 1.0301× | 99.94%吞吐 |
| 计PyTorch copy提交 | 598.256 ms | 1.0202× | 98.97%吞吐 |

第二行仍未包含所有新增Python callback和compute-stream Event，因此只是
保守方向的估算，不是实测结果。当前会话中PyTorch报告
`cuda_available=false`，无法生成新的GPU时间线。正式结论必须在GPU
恢复可见并取得真实checkpoint后重跑。

当前可以确定的是：

- 显存节省来自最大slot从416 MiB下降到112 MiB；
- BF16 H2D payload没有减少；
- 相对整层双缓冲的性能预期是基本持平到约1%回退；
- 相对单缓冲串行仍保留约2%～3%的流水收益；
- 真正提高Token/s仍需后续INT8/INT4传输和GPU融合反量化。

## 正确性和运行验证

已通过8项CPU侧单元测试：

- 默认矩阵粒度及224个传输单元；
- 整层回退及32个传输单元；
- 112 MiB自动slot和224 MiB双slot；
- 独立/绑定LM Head两种resident布局；
- 608 MiB显存节省计算；
- `full_pinned`分配策略；
- `pinned_staging`的pageable主存、两个pinned slot；
- staging slot等待H2D Event后再复用。

同时通过所有新增Python文件的字节码编译。GPU执行器、真实H2D时间、
logits一致性和端到端生成仍需CUDA设备与Meta授权checkpoint。

结构化计算结果见
[`results/llama31_8b_matrix_runtime_report.json`](results/llama31_8b_matrix_runtime_report.json)。
