# Llama-3.1-8B CPU词表流式化真实实验

## 实现范围

- Transformer每层分为QKV、O、Gate/Up、Down四个完整矩阵组；
- 不切单个矩阵内部Tile，不使用自定义算子；
- Embedding根据Token ID从CPU `[V,H]` 行连续布局提取所需行；
- LM Head按词表行切成8个约128 MiB块，每块16,384行，末块13,568行；
- 两个词表矩阵共享CPU布局和调度实现，但Llama-3.1-8B
  `tie_word_embeddings=false`，数值权重保持独立；
- LM Head使用标准`F.linear`和`torch.topk`在线归并全局Top-10。

## 结果

| 配置 | 中位ms/token | token/s | CUDA peak allocated |
|---|---:|---:|---:|
| 上一版本：matrix、词表常驻 | 582.340 | 1.717 | 2.197 GiB |
| matrix_group、词表常驻 | 581.660 | 1.719 | 2.416 GiB |
| matrix_group、词表流式 | 624.162 | 1.602 | 0.448 GiB |

相对上一版本，词表流式版本：

- CUDA peak allocated降低 **4.90倍**，
  节省 **79.60%**、
  1.749 GiB；
- 延迟增加 **7.18%**，
  保留 **93.30%** 吞吐；
- P10–P90为
  624.076–624.210 ms；
- 计划GPU权重区为
  448.508 MiB，实测峰值为
  459.069 MiB。

矩阵分组本身相对上一版本的延迟变化只有
-0.117%，新增延迟主要来自
LM Head流式传输。

## LM Head诊断

每Token新增传输1.051 GB：

- 8次H2D累计43.702 ms；
- H2D有效吞吐24.042 GB/s；
- 分块GEMV和在线Top-k累计
  2.184 ms；
- LM Head流水总计44.057 ms。

Embedding在单Token decode只传输8 KiB，不再完整加载约0.979 GiB矩阵。

## 正确性

连续8个greedy token全部匹配CPU参考。完整`[8,128256]`词表logits最低
cosine为0.999739，8步argmax
全部匹配；误差与上一版BF16专用执行器处于同一范围。在线Top-k由每块局部
Top-k和全局候选再次Top-k构成，不使用近似词表裁剪。
