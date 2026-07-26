# Llama-3.1-8B 单卡 Cascade 与 AirLLM 实测对比

## 结论

在 RTX 3080 Ti 单卡、BF16、batch=1、6-token prompt、1 次 decode 预热和
7 次稳态 decode 的统一口径下，Cascade 推荐配置的稳态速度是 AirLLM
热切分复测的 **11.26 倍**，prefill 是 **10.15
倍**。AirLLM 的峰值 CUDA allocated 更低：Cascade 是 AirLLM 的
**2.22 倍**，即 AirLLM 少用 **55.01%**、
约 **1.209 GiB**。

| 指标 | AirLLM 3.0.1 | Cascade |
|---|---:|---:|
| 配置 | BF16、默认预取、无压缩 | full_pinned、matrix、2 slots |
| 稳态中位延迟 | 6555.016 ms/token | 582.340 ms/token |
| 稳态速度 | 0.153 token/s | 1.717 token/s |
| P10–P90 | 6534.794–6568.603 ms | 582.281–582.413 ms |
| Prefill | 7381.899 ms | 727.397 ms |
| CUDA peak allocated | 0.989 GiB | 2.197 GiB |
| CUDA peak reserved | 1.002 GiB | 2.223 GiB |
| 测量窗口物理磁盘读取 | 0 B | 0 B |

## 为什么速度相差较大

AirLLM 每 token 流式处理全部 35 个单元，共
16.061 GB。单次诊断中，权重读取/映射累计
4988.721 ms，权重安装/H2D CUDA Event 累计
1442.096 ms，H2D 有效吞吐
11.137 GB/s；实际输入 H2D 的 pinned
权重为 0 B。读取阶段虽然命中 Linux page
cache、没有产生物理磁盘读取，仍包含逐层 safetensors 映射、CPU 张量和
Python/内存管理成本。诊断累计时间来自不同线程，不能作为严格互斥的
wall-time 分解。

Cascade 每 token 传输 13.959 GB，
H2D Event 累计 580.558 ms，有效吞吐
24.044 GB/s；计算 Event 累计
24.762 ms。Embedding、LM Head 和小型 Norm
常驻 GPU，因此传输量比 AirLLM 少；完整 CPU 权重一次性锁页，热路径不再
逐层读取或动态 pin/unpin。

## 显存与主存权衡

AirLLM 会流式加载 Embedding、32 层、Norm 和 LM Head，因此峰值主要由
最大单层/Embedding 决定，只需 0.989 GiB CUDA allocated。
Cascade 将 Embedding、LM Head 和 Norm 常驻，并预分配两个矩阵 slot，
因此是 2.197 GiB，但换来显著更高吞吐。

Cascade 的 full_pinned 模式锁页 14.958
GiB CPU 权重。AirLLM 不把全量权重常驻进程内锁页；它依赖按层文件和 OS
page cache，已有切分文件占 14.958 GiB
额外 SSD 空间。首次创建切分并初始化为
17.509 秒，复用切分时为
1.371 秒。

## 正确性与限制

两者前 8 个 greedy token ID 均为 `[311, 1505, 701, 8352, 13, 578, 7580, 315]`，与已有 Transformers
CPU 参考一致。两边使用同一 checkpoint、同一 GPU 和 BF16，但 Transformers
版本分别为 4.57.6 与
4.45.2；Cascade 使用专用 Llama 执行器，
AirLLM 使用 Hugging Face 通用模型及逐层 hook。结果是本机单请求短上下文
decode 数据，不代表长上下文、批处理、量化或冷 page-cache 场景。
