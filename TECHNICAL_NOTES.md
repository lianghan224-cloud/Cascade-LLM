# 技术测量笔记：Llama-3.2-1B CPU 常驻 / GPU 异步分层推理

这是最终 HTML 报告的可读源材料与方法细节；主要交付见
`report.html`。

测试日期：2026-07-25。目标机器：`csu01`。

## 结论

这台机器具备实现真正 H2D/计算重叠的硬件条件：两张 RTX 3080 Ti
都报告 2 个 async copy engine，单卡 PCIe Gen4 x16，且全模型 BF16
权重可以一次性放入 page-locked CPU 内存。

关键结果如下：

- CUDA Driver API 的 pinned H2D 上限为 GPU0 `24.24 GB/s`、GPU1
  `24.06 GB/s`。一个精确 decoder layer（含两个 norm，`116.008 MiB`）
  分别需要 `5.020 ms`、`5.055 ms`。
- 单次 pinned async H2D 的 host enqueue 中位数约 `1.8 µs`。1 B/4 KiB
  批量 tiny-copy 的稳态 command slope 为 `1.82–2.13 µs/copy`；0 B
  会被优化成 no-op，不能用来代表 DMA 启动成本。
- pageable CPU 内存的 116 MiB “async”调用会在 host 侧阻塞约 `9.1 ms`，
  只有约 `13.3 GB/s`，不能用于真正重叠。
- 精确形状、BF16、projection-only 的单层计算在 `M=batch×tokens`
  为 1/512/2048/4096 时，GPU0 分别约为
  `0.212/1.286/4.035/7.733 ms`。整层 H2D 与计算的交点约在
  `M≈2.5k–2.7k`。
- 16 层、两个真实 device weight slot 的流水测试中，GPU0 在
  `M=1/512/1024/2048/4096` 的加速分别是
  `1.033×/1.200×/1.373×/1.712×/1.587×`；GPU1 为
  `1.031×/1.200×/1.373×/1.713×/1.586×`。

所以这个方向是可行的，但收益高度依赖工作负载。单 token、小 batch
decode 时，GPU 读权重比 PCIe 搬权重快约一个数量级以上，流水只能填掉
很小的计算气泡，稳态仍由 H2D 总字节数决定。prefill 或大 batch 达到
约 2.6k rows 后，整层 copy 基本可以被当前层计算隐藏。

## 当前测试边界

[官方模型仓库](https://huggingface.co/meta-llama/Llama-3.2-1B)是
manual-gated；本机当前没有 Hugging Face token，也没有已缓存的权重，
权重下载返回 401。因此本轮没有声称完成真实模型的端到端推理或 logits
正确性验证。

模型元数据、精确 tensor 数量和形状可以从公开 Hub API 得到；GPU 计算
使用这些精确 projection 形状和 BF16 dtype 的合成权重。它包含
q/k/v/o、gate/up/down 与 SwiGLU，但刻意不含 RMSNorm、RoPE、
QK-softmax-V、KV-cache 读写和 residual。16-stage 流水的 copy 目标
就是随后 compute 使用的两个 slot，不是只做了无关 buffer 的并发拷贝。

## 基础环境

| 项目 | 本机 |
|---|---|
| OS / kernel | Ubuntu 20.04.4 / Linux 5.15.0-139 |
| CPU | AMD Threadripper 3970X，32C/64T，单 NUMA，128 MiB L3 |
| RAM | 125.6 GiB，总体约 120 GiB 可用 |
| GPU | 2 × RTX 3080 Ti 12 GiB，compute capability 8.6 |
| PCIe | 两卡均最大 Gen4 x16；两卡间 `SYS`，无 NVLink |
| Driver / runtime | NVIDIA 555.42.02；Driver API 12.5；PyTorch CUDA 12.4 |
| Python stack | Python 3.8.10；PyTorch 2.4.1+cu124；Transformers 4.45.2 |
| 存储 | 当前 workspace `/disk2` 是 SATA HDD；`/ssd` 是 NVMe |

正式运行应把 checkpoint 放在 `/ssd`，但稳态推理按本设计只从 CPU RAM
读取，磁盘只影响启动阶段。

## Llama-3.2-1B 权重布局

Hub 当前 revision 为
`4e20de362430cd3b72f300e6b0f18e50e7166e08`；
`model.safetensors` 文件为 `2,471,645,608 B`，其中 tensor payload
为 `1,235,814,400` 个 BF16 参数，即 `2,471,628,800 B`。
[官方 model card](https://huggingface.co/meta-llama/Llama-3.2-1B)
也确认模型约 1.23B 参数、128k context、GQA 和 tied/shared embedding。

| 组成 | BF16 字节 |
|---|---:|
| tied embedding / lm_head | 525,336,576 B = 501 MiB |
| 单 decoder layer（含两个 norm） | 121,643,008 B = 116.008 MiB |
| 16 个 decoder layer | 1,946,288,128 B = 1,856.125 MiB |
| final norm | 4,096 B |
| 全部 tensor | 2,471,628,800 B = 2.302 GiB |

单层自然矩阵边界为：K/V 各 2 MiB，Q/O 各 8 MiB，attention 合计
20 MiB，gate/up/down 各 32 MiB，MLP 合计 96 MiB。每层两个 norm
合计 8 KiB。

GQA KV cache（BF16、16 层、8 KV heads、head dim 64）为
`32,768 × batch × context` 字节：batch 1 时，8k/32k/128k context
约为 256 MiB/1 GiB/4 GiB。

这个 1B 模型本身能完整放入 12 GiB GPU；它适合作为“强制 out-of-core”
的受控实验，并可在同一张卡上运行 full-resident reference 做正确性和
性能对照。

## H2D 与 CPU 内存结果

下表是直接调用 `cuMemcpyHtoDAsync_v2` 的 median，不含 PyTorch eager
dispatch：

| 分片 | GPU0 时间 / GB/s | GPU1 时间 / GB/s |
|---:|---:|---:|
| 64 KiB | 5.344 µs / 12.26 | 5.376 µs / 12.19 |
| 1 MiB | 42.592 µs / 24.62 | 42.656 µs / 24.58 |
| 2 MiB | 82.816 µs / 25.32 | 82.720 µs / 25.35 |
| 8 MiB | 321.568 µs / 26.09 | 321.696 µs / 26.08 |
| 20 MiB | 799.328 µs / 26.24 | 799.744 µs / 26.22 |
| 32 MiB | 1.277 ms / 26.27 | 1.278 ms / 26.25 |
| 96 MiB | 4.144 ms / 24.29 | 4.197 ms / 23.98 |
| 116.008 MiB | 5.020 ms / 24.23 | 5.055 ms / 24.07 |
| 256 MiB | 11.040 ms / 24.32 | 11.062 ms / 24.27 |

以 `1.8 µs` 固定 command/enqueue 成本和约 `24 GB/s` 估算，固定成本低于
总时间 5% 需要约 0.8 MiB，低于 1% 需要约 4.1 MiB。实际调度建议最小
分片为 1–2 MiB，并优先沿模型已有矩阵边界切分。

全模型大小的 `cuMemHostAlloc(2,471,628,800 B)` 已成功：

- 分配 `1.403 s`；
- 单次 libc `memset` 首次完整触页 `0.175 s`，约 `14.14 GB/s`；
- `cuMemFreeHost` 为 `0.596 s`。

如果权重先放在普通 pageable RAM，再由单 CPU thread 搬入 pinned
staging，116 MiB memcpy 为约 `8.2 ms / 14.8 GB/s`，比 H2D 本身更慢。
因此 1B 首版应直接使用全量 pinned CPU 权重；扩展到超大模型时再采用
pageable store + 多线程 pinned staging ring，并单独校准 CPU DRAM
吞吐。

## GPU 计算与重叠

GPU0 单 decoder layer projection-only：

| M | 计算时间 | 有效 TFLOP/s | 116 MiB nominal-weight/time |
|---:|---:|---:|---:|
| 1 | 0.212 ms | 0.574 | 573.8 GB/s |
| 128 | 0.383 ms | 40.7 | 317.7 GB/s |
| 512 | 1.286 ms | 48.4 | 94.6 GB/s |
| 1024 | 2.075 ms | 60.0 | 58.6 GB/s |
| 2048 | 4.035 ms | 61.7 | 30.1 GB/s |
| 4096 | 7.733 ms | 64.4 | 15.7 GB/s |

两个 persistent CUDA stream、两个预分配 weight slot、16 个 stage：

| M | GPU0 串行 | GPU0 重叠 | 加速 | GPU1 加速 |
|---:|---:|---:|---:|---:|
| 1 | 86.571 ms | 83.839 ms | 1.033× | 1.031× |
| 128 | 86.558 ms | 81.744 ms | 1.059× | 1.059× |
| 512 | 98.960 ms | 82.458 ms | 1.200× | 1.200× |
| 1024 | 114.452 ms | 83.344 ms | 1.373× | 1.373× |
| 2048 | 146.023 ms | 85.292 ms | 1.712× | 1.713× |
| 4096 | 205.652 ms | 129.546 ms | 1.587× | 1.586× |

理想 N-stage 时间近似为：

```text
serial  = N × (D + C)
overlap = D + (N - 1) × max(D, C) + C
```

分片和流水只能去掉 `D+C` 中较短的一侧，不能减少每 token 必须经过
PCIe 的总权重字节。若 embedding/head 常驻 GPU，只流 16 个 decoder
layer，则单 token 的 H2D 下界约为 `80.3 ms`，即约 `12.45 token/s`
（还没有计入 lm_head、attention、Python 和调度开销）。

## 大分片实测：1/2/4/8 层组成一个连续分片

为避免反复读取同一层导致结果偏乐观，新基准预先分配完整 16 层、
`1856.125 MiB` 的 pinned CPU arena。每个 stage 从不同的连续区间搬运，
总 H2D 字节和 projection 计算量固定，只改变每片包含的层数。GPU0
按 `1→2→4→8` 测试，GPU1 按 `8→4→2→1` 反向测试；下表是两张卡各自
median 的均值。

纯 Driver API pinned async H2D：

| 每片层数 | 分片大小 | H2D 时间 | 带宽 | host call |
|---:|---:|---:|---:|---:|
| 1 | 116.008 MiB | 5.051 ms | 24.091 GB/s | 1.818 µs |
| 2 | 232.016 MiB | 10.115 ms | 24.059 GB/s | 1.829 µs |
| 4 | 464.031 MiB | 20.230 ms | 24.054 GB/s | 1.848 µs |
| 8 | 928.062 MiB | 40.358 ms | 24.113 GB/s | 2.064 µs |
| 16 | 1856.125 MiB | 80.750 ms | 24.103 GB/s | 2.128 µs |

116 MiB 到 1856 MiB 的带宽没有继续提高，始终约为 `24.1 GB/s`。
也就是说，116 MiB 单层早已处于 PCIe 平台区；把多个整层合并只会让
H2D 时间近似线性增长。host call 虽从约 1.82 µs 增到 2.13 µs，但相对
5–81 ms 的 DMA 时间可以忽略。

固定 16 层的双 slot 流水：

| 每片层数 | 双 slot 权重 | M=1 加速 | M=512 加速 | M=2048 加速 | M=4096 加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 232.016 MiB | 1.033× | 1.192× | 1.674× | 1.582× |
| 2 | 464.031 MiB | 1.030× | 1.178× | 1.605× | 1.521× |
| 4 | 928.062 MiB | 1.025× | 1.148× | 1.477× | 1.415× |
| 8 | 1856.125 MiB | 1.017× | 1.094× | 1.275× | 1.242× |

关键工作量的绝对吞吐：

| 每片层数 | M=2048 overlap | 相对 g=1 吞吐 | M=4096 overlap | 相对 g=1 吞吐 |
|---:|---:|---:|---:|---:|
| 1 | 87.143 ms | 100.0% | 130.062 ms | 100.0% |
| 2 | 90.928 ms | 95.8% | 135.196 ms | 96.2% |
| 4 | 98.759 ms | 88.2% | 145.747 ms | 89.2% |
| 8 | 114.427 ms | 76.2% | 166.165 ms | 78.3% |

原因是分组会减少 pipeline stage 数，放大填充/排空损失。设每层加载
和计算时间分别为 `D`、`C`，共 `L=16` 层，每片 `g` 层，忽略冲突时：

```text
T_serial(g)  ≈ L × (D + C)
T_overlap(g) ≈ L × max(D, C) + g × min(D, C)
```

因此固定总工作量下，`g` 增大不会带来正比加速；第二项反而随 `g`
线性增长。减少 H2D/event 提交次数只节省几十微秒，而 M=2048 从
`g=1` 增到 `g=8` 多出约 `27.3 ms` 的填充/排空时间。实测与无竞争理想
模型很接近：M=2048 的观测/理想加速分别从 `1.674×/1.711×` 降到
`1.275×/1.286×`；M=4096 从 `1.582×/1.587×` 降到
`1.242×/1.246×`。

结论不是“分片越小越好”，而是分片应小到提供足够 stage、又大到饱和
H2D 并避免过多 kernel/event/bookkeeping。本机从 1–2 MiB 已接近带宽
平台；若调度单位必须是完整 decoder layer，则 1 层/片明显优于把
2/4/8 层合成大分片。若要进一步逼近 `T_copy ≈ T_compute`，应向层内的
2/8/32 MiB 自然 projection 边界拆分，而不是向多层合并。

原始数据见 `results/h2d_large_shards_gpu{0,1}.json` 和
`results/large_shard_pipeline_gpu{0,1}.json`；两卡反向顺序复现、
公式重算、SHA-256 与百分位检查共 221 项，全部通过，汇总见
`results/large_shards_summary.json`。

## AirLLM 当前实现审查

审查版本：
[`17677cb`](https://github.com/lyogavin/airllm/commit/17677cb821016b36a0610c8e1f2befab030d1942)。

- 当前 prefetch worker 只提前做 disk/safetensors → CPU；
  [`_pre_hook`](https://github.com/lyogavin/airllm/blob/17677cb821016b36a0610c8e1f2befab030d1942/air_llm/airllm/airllm_base.py#L352-L367)
  先等待 CPU future，再同步地逐 tensor 移到 GPU，之后才启动下一层的
  CPU prefetch。因此真正的 H2D 没有和当前层 GPU compute 重叠。
- `state_dict[k].pin_memory()` 的返回值没有写回；PyTorch 的
  `pin_memory()` 不是 in-place，所以原 state dict 仍是 pageable。
- H2D 使用逐 tensor `set_module_tensor_to_device`，没有独立 copy
  stream、`non_blocking` copy、ready/free event 或固定 device slot。
- post-hook 每层执行 `module.to("meta")`，随后 `gc.collect()`、
  `malloc_trim()` 和 `torch.cuda.empty_cache()`，会产生 allocator
  churn，并破坏稳定流水。
- tied embedding/head 被常驻 GPU，这一点对 Llama-3.2-1B 是合理的，
  固定占用 501 MiB。

所以无需沿用 AirLLM 的 hook + meta allocator 路径。保留它的
checkpoint 拆分/模型兼容思路即可，运行时更适合做独立 scheduler。

## 建议的首版框架

```text
PinnedWeightStore (CPU, 2.302 GiB)
             │
             ├── copy stream ──> device slot[0/1] ── ready event ─┐
             │                                                     │
             └──────── free/reuse event <── compute stream <───────┘
```

1. 启动时解析 safetensors header，按逻辑 layer/tensor 建 offset manifest，
   直接把文件读入一个连续 pinned arena。不要依赖文件中的字典顺序。
2. GPU 常驻 tied embedding/head、所有 norm、RoPE buffer、KV cache；
   权重使用两个预分配 slot。整层 slot 只需额外 `232.016 MiB`。
3. copy stream 写 slot B，同时 compute stream 读 slot A。copy 完成记录
   `ready[i]`；compute 等待 ready，结束记录 `free[i]`；slot 覆盖前只等
   对应 free event。整次 forward 尾部才同步，不逐层 synchronize。
4. 不在热路径做 `module.to()`、分配/释放、`empty_cache()` 或 CPU
   `pin_memory()`。执行层应通过预绑定 view 或 functional kernel 直接读取
   slot。
5. 运行时 autotuner 按 `(M, context, dtype, GPU)` 记录
   `T_compute` 和 `T_H2D`，用
   `S_prefetch ≈ B_H2D × T_compute(window)` 选择下一次可以隐藏的字节数。

建议粒度：

| M | 一个整层计算窗口可隐藏的约当 H2D | 建议 |
|---:|---:|---|
| 1 | 约 5 MiB | 2/8 MiB tensor 粒度只能改善首片延迟；总体仍 H2D-bound |
| 512 | 约 29 MiB | 32 MiB MLP matrix 是自然边界 |
| 2048 | 约 92 MiB | 96 MiB MLP 或整层 |
| 4096 | 约 177 MiB | 116 MiB 整层即可完全隐藏 |

为了进一步降低首片 latency，可做 intra-layer schedule：先加载并计算
Q/K/V，copy stream 同时搬 O 与 MLP；attention 执行时继续搬
gate/up/down。若要切到单个矩阵内部，则必须实现 output/input tiling，
不能只切 safetensors 后仍调用完整 `F.linear`。

[PyTorch 官方说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
也给出了设备侧真正重叠的三个必要条件：可用 DMA engine、独立非默认
stream、pinned source。本机三项均已满足。

## 下一步

1. 在本机完成 Hugging Face license 接受并通过环境变量或
   `huggingface-cli login` 提供 token；不要把 token 写进仓库。
2. 下载到 `/ssd`，生成 pinned arena/offset manifest。
3. 实现 Llama decoder 的整层双缓冲版本，先与 full-resident reference
   比较每层输出与最终 logits。
4. 再实现 projection 粒度 schedule，测试 batch 1 decode、不同 batch
   和 128–4096 token prefill。
5. 用 CUDA Event 和 Nsight Systems 同时检查 copy engine、kernel、
   slot lifetime、峰值 VRAM、TTFT 和 token/s。

原始数据和复现方法见 [README.md](README.md)。
