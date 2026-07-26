# Cascade-LLM：CPU常驻权重的矩阵级流式推理

> Active real-model branch: `real-llama31-8b-experiment`

Cascade-LLM面向个人用户的单机、单GPU、单请求推理：完整模型权重保存在
CPU内存，只把即将执行的权重矩阵异步传入GPU，在降低显存需求的同时尽量
让H2D与当前矩阵计算重叠。本分支以
`meta-llama/Llama-3.1-8B` BF16为第一版目标，思路来源于AirLLM，但运行时
不使用逐层hook、`module.to("meta")`和热路径动态分配。

模型权重受Meta许可约束，仓库不包含checkpoint。运行真实生成前，需要用户
自行接受模型许可并下载到本地。

## 真实 Llama-3.1-8B 实验结果

真实实验使用ModelScope
`LLM-Research/Meta-Llama-3.1-8B` revision
`39ef6178f1f193f3636d4f6150d6966dc05fa366`。4个safetensors包含291个
BF16 Tensor、8,030,261,248个参数，参数payload为14.958 GiB。运行时从
CPU全量权重执行真实RMSNorm、RoPE、GQA Attention/SDPA、MLP和残差计算，
不是合成GEMM。

本机环境为Threadripper 3970X、125 GiB RAM、RTX 3080 Ti 12 GiB、
PCIe Gen4 x16。测试口径为batch=1、6-token prompt、1次decode warmup、
每配置7次无细粒度profiling的单Token wall-time样本；P10/P90由这7次
样本计算。另运行3次CUDA Event profile诊断H2D和计算，但不把重叠流上的
event总和当成严格wall-time分解。

| CPU权重模式 | 粒度 | GPU slot | 中位ms/token | P10–P90 ms | token/s | 权重显存 |
|---|---|---:|---:|---:|---:|---:|
| full_pinned | 层 | 1 | 603.385 | 603.341–603.481 | 1.657 | 2.364 GiB |
| full_pinned | 矩阵 | 1 | 605.715 | 605.622–605.783 | 1.651 | 2.067 GiB |
| full_pinned | 层 | 2 | **582.235** | 582.205–582.371 | **1.718** | 2.770 GiB |
| full_pinned | 矩阵 | 2 | 583.031 | 582.978–583.110 | 1.715 | **2.176 GiB** |
| pinned_staging | 层 | 2 | 1129.806 | 1125.983–1130.434 | 0.885 | 2.770 GiB |
| pinned_staging | 矩阵 | 2 | 1231.850 | 1229.532–1233.287 | 0.812 | 2.176 GiB |

结论：

- 推荐默认配置是`full_pinned + matrix + 2 slots`。它比最快的层粒度
  双缓冲只慢0.137%，但少用608 MiB权重显存。
- 该配置的计划权重显存为2.176 GiB，相对完整BF16参数载荷缩减
  **6.873倍**，即节省85.45%；实际CUDA peak allocated为2.197 GiB。
- 相同矩阵粒度下，双缓冲相对单缓冲只加速**1.039倍**。每Token传输
  13,958,643,712 bytes，H2D约24.0 GB/s；主配置H2D/Compute Event
  中位总和分别为581.238/24.636 ms，系统明显受H2D限制。
- `full_pinned`完整锁页14.958 GiB CPU权重；`pinned_staging`矩阵双slot
  只锁页224 MiB，但延迟慢至2.113倍。因此前者适合专用推理机，后者是
  锁页内存受限时的兼容模式。
- decode测量窗口的进程磁盘读取增量在所有8组测试中均为0，SSD不在热
  路径；GPU1复测主配置与GPU0的中位延迟只差0.222%。
- 与Transformers CPU参考连续比较8个greedy token，token ID 8/8一致，
  最低完整词表logit cosine为0.999739。

普通full-GPU基线无法在本机运行：仅14.958 GiB参数payload就已经超过
11.668 GiB物理显存，还未包含KV Cache、activation和workspace。因此本
报告只给出相对普通基线的权重显存倍数，不虚构full-GPU速度倍数。

详细方法、原始数据、图表、限制和后续实验见自包含报告
[real_results/report.html](real_results/report.html)；机器可读汇总和
10项QA收据见
[real_results/real_llama31_summary.json](real_results/real_llama31_summary.json)。

### 真实基准复现

```bash
source scripts/activate_env.sh

CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/real_llama31_benchmark.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B \
  --weight-store full_pinned \
  --granularity matrix \
  --slots 2 \
  --warmup-decode 1 \
  --decode-repeats 7 \
  --profile-repeats 3 \
  --output real_results/bench_full_pinned_matrix_s2.json

.venv/bin/python benchmarks/summarize_real_llama31.py
.venv/bin/python benchmarks/build_real_llama31_artifact.py
```

### AirLLM公平对照环境

AirLLM使用独立Python 3.11环境，避免其`transformers>=4.49`依赖修改
Cascade的真实实验环境。安装脚本主动清除大小写代理变量、忽略用户pip
配置、固定AirLLM源码commit，并复用现有Llama-3.1-8B权重：

```bash
cd /disk2/home/guest/lianghan/repos/Cascade-LLM
bash scripts/install_airllm_no_proxy.sh
```

默认持久化位置：

```text
/ssd/cascade-llm/third_party/airllm   # 固定commit的源码
/ssd/cascade-llm/venvs/airllm        # 独立虚拟环境
/ssd/cascade-llm/python              # uv管理的Python 3.11
real_results/airllm/install_receipt.json
```

安装完成后激活：

```bash
source scripts/activate_airllm_env.sh
```

如果官方PyTorch CDN过慢，可以中断当前AirLLM安装并切换到阿里云PyPI和
PyTorch CUDA 12.1镜像。默认保留已完成的uv缓存，只删除未完成临时文件：

```bash
bash scripts/restart_airllm_cn_mirror.sh --reuse-cache
```

如确认不需要已有包缓存，可只清理AirLLM虚拟环境和包缓存后重新下载；该
命令不会删除模型权重和AirLLM源码：

```bash
bash scripts/restart_airllm_cn_mirror.sh --clean
```

安装步骤不拆分模型也不启动正式推理。冷缓存/热缓存性能和峰值显存对照应
使用后续统一benchmark脚本，不能把AirLLM初始化拆分时间计入单Token
decode口径。

### AirLLM 8B 单卡真实对比

在GPU0上使用同一Llama-3.1-8B BF16 checkpoint、同一6-token prompt、
batch=1、1次decode warmup和7次稳态decode进行对比。AirLLM为3.0.1、
默认预取且不压缩；Cascade为`full_pinned + matrix + 2 slots`。

| 指标 | AirLLM 3.0.1 | Cascade |
|---|---:|---:|
| 稳态中位延迟 | 6555.016 ms/token | 582.340 ms/token |
| 稳态速度 | 0.153 token/s | 1.717 token/s |
| P10–P90 | 6534.794–6568.603 ms | 582.281–582.413 ms |
| Prefill | 7381.899 ms | 727.397 ms |
| CUDA peak allocated | 0.989 GiB | 2.197 GiB |
| CUDA peak reserved | 1.002 GiB | 2.223 GiB |

Cascade在该口径下是AirLLM的**11.26倍**速度，但CUDA peak allocated是
AirLLM的2.22倍，多用约1.209 GiB。AirLLM逐token传输全部35个单元；
诊断得到权重读取/映射累计4988.721 ms，H2D/参数安装累计1442.096 ms，
有效H2D吞吐11.137 GB/s，进入H2D的pinned权重实测为0 B。Cascade把
Embedding、LM Head和Norm常驻，只传输Transformer矩阵，H2D累计
580.558 ms、有效吞吐24.044 GB/s。

两者在测量窗口的物理磁盘读取均为0 B，前8个greedy token完全一致。
AirLLM的低显存来自逐层流式加载所有模块；Cascade多用显存换取常驻输出层、
预分配双slot和全量CPU锁页权重。完整报告见
[`real_results/airllm/comparison_report.md`](real_results/airllm/comparison_report.md)。

复现AirLLM：

```bash
CUDA_VISIBLE_DEVICES=0 \
  /ssd/cascade-llm/venvs/airllm/bin/python \
  benchmarks/airllm_llama31_8b_benchmark.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B \
  --layer-shards-root /ssd/cascade-llm/airllm-layer-shards/llama31-8b \
  --warmup-decode 1 \
  --decode-repeats 7 \
  --profile-repeats 1 \
  --output real_results/airllm/bench_airllm_bf16_prefetch_gpu0.json

.venv/bin/python benchmarks/summarize_airllm_comparison.py
```

## 历史合成硬件校准

`results/`和仓库根目录的旧H2D数据是开发真实运行时之前的合成硬件校准，
不能与`real_results/`的端到端真实模型结果混淆。它们仍用于估算固定H2D
提交开销和检查两张GPU的一致性；真实结论一律以上一节为准。

## 真实实验的持久化环境

源码仓库、模型权重和下载缓存相互分离：

```text
/disk2/home/guest/lianghan/repos/Cascade-LLM   # Git源码和可提交收据
/ssd/cascade-llm/models                       # 受限checkpoint，不进Git
/ssd/cascade-llm/hf-cache                     # Hugging Face持久缓存
/ssd/cascade-llm/uv-cache                     # Python包持久缓存
```

加载本地路径配置：

```bash
source scripts/activate_env.sh
```

该配置默认使用阿里云PyPI镜像安装Python依赖，并把包缓存持久化到
`/ssd/cascade-llm/uv-cache`；模型权重仍从Hugging Face官方受限仓库下载。

检查GPU、Python、认证和checkpoint状态：

```bash
python scripts/check_real_environment.py
```

接受Meta许可并执行`huggingface-cli login`后，下载固定revision的真实权重：

```bash
python scripts/download_llama31_8b.py
```

网络受限时，也可以从ModelScope的同模型BF16镜像下载。镜像下载器固定
ModelScope commit，并按公开manifest逐文件验证大小和SHA-256：

```bash
python scripts/download_llama31_8b_modelscope.py
```

下载脚本不会把Token或权重写入Git，只会在`real_results/`生成可提交的
revision、文件大小和SHA-256收据。完整验收顺序见
[`REAL_EXPERIMENT_PROTOCOL.md`](REAL_EXPERIMENT_PROTOCOL.md)。

## Llama-3.1-8B 原型运行时

`layer_streaming/` 现在包含面向单机单请求的Llama-3.1-8B运行时：

- 默认按单个Projection矩阵进行H2D；
- 根据最大矩阵自动分配两个GPU weight slot；
- `full_pinned`完整锁页CPU权重模式；
- `pinned_staging` pageable主存加两个pinned staging slot模式；
- Embedding、LM Head和Norm使用独立GPU常驻区；
- Copy/Compute两个CUDA Stream及ready/free Event流水。

查看静态计划和基线对比：

```bash
.venv/bin/python benchmarks/report_llama31_matrix_runtime.py
```

输出包括整层/矩阵粒度的自动slot、两种CPU锁页模式、相对不同基线的显存
比例，以及基于已保存8B实测数据的性能区间和限制。

真实checkpoint生成入口：

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B \
  --weight-store full_pinned \
  --granularity matrix \
  --prompt "The meaning of life is" \
  --max-new-tokens 8
```

兼容模式：

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint /ssd/cascade-llm/models/Llama-3.1-8B \
  --weight-store pinned_staging \
  --granularity matrix
```

运行单元测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

实现范围、显存节省和性能口径见
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md)。

本目录包含 Llama-3.2-1B“CPU 权重常驻、GPU 双 slot、H2D 与计算重叠”
的本机校准工具和原始结果。主要交付是自包含的
[report.html](report.html)；更详细的代码/方法笔记见
[TECHNICAL_NOTES.md](TECHNICAL_NOTES.md)。

这个目录给“CPU 常驻权重、分片异步搬运、GPU 计算与下一片 H2D
重叠”的调度器提供第一组本机基线。基准程序不需要 CUDA Toolkit、
CUDA 头文件、PyTorch 或 CUDA Runtime；它只在运行时
`dlopen("libcuda.so.1")` 并调用 CUDA Driver API。

## 文件

- `h2d_driver_bench.c`：独立 C11 基准程序。
- `h2d_results_gpu0.json`、`h2d_results_gpu1.json`：两张卡的完整原始结果，
  每个点均含 warmup 次数、样本数、p10/median/p90。
- `h2d_results_summary.json`：本机、关键模型尺寸、线性拟合与原始文件
  SHA-256 的紧凑摘要。
- `benchmarks/torch_llama32_1b_bench.py`：精确 Llama projection 形状、
  copy/compute 四组对照和 16-stage 双 slot 流水。
- `benchmarks/large_shard_pipeline_bench.py`：固定 16 层总工作量，比较
  1/2/4/8 层组成一个连续分片时的双 slot 流水。
- `benchmarks/validate_large_shards.py`：重算大分片结果、跨卡一致性、
  单调趋势与相对吞吐。
- `benchmarks/summarize_shard_accounting.py`：汇总 M=1 时不同分片数量的
  H2D、GPU计算、Driver/PyTorch提交和端到端时间。
- `benchmarks/llama31_8b_m1_bench.py`、`summarize_llama31_8b.py`：
  Llama-3.1-8B 精确层形状的 M=1、8 GiB Arena 双卡基准与校验汇总。
- `benchmarks/pinned_arena_probe.py`：用 CUDA Driver API 分配并触碰一个
  与完整 BF16 checkpoint 相同大小的 pinned CPU arena。
- `benchmarks/validate_results.py`：重算 SHA、H2D median、流水公式、
  hidden-fraction 口径与两卡一致性。
- `benchmarks/build_report_artifact.py`：从已保存 JSON 生成规范化
  `artifact.json` 和报告 source notes。
- `results/torch_llama32_1b_gpu{0,1}.json`：两卡 PyTorch/流水原始结果。
- `results/h2d_large_shards_gpu{0,1}.json`：116.008 MiB 至
  1856.125 MiB 的 pinned async Driver H2D 原始结果。
- `results/large_shard_pipeline_gpu{0,1}.json`：两卡多层分组流水结果。
- `results/large_shards_summary.json`：大分片两卡均值、SHA-256 和
  221 项 QA receipt。
- `results/shard_accounting_m1_gpu{0,1}.json`、
  `results/shard_accounting_m1_summary.json`：单Token点的组件时间账本。
- `results/llama31_8b_m1_gpu{0,1}.json`、
  `results/llama31_8b_m1_summary.json`：8B 双卡组件与流水结果。
- `results/h2d_llama31_8b_shards_gpu{0,1}.json`：8B 精确层倍数的
  Driver API pinned H2D 原始结果。
- `results/pinned_arena_probe.json`：2.302 GiB 全量 pinned arena 结果。
- `results/validation.json`：65 项结果 QA receipt。
- `report.html`：自包含、离线可读的最终技术报告。

## 构建和复现

```bash
cc -O2 -std=c11 -Wall -Wextra -Werror -pedantic \
  h2d_driver_bench.c -o h2d_driver_bench -ldl -lm

./h2d_driver_bench --device 0 --output h2d_results_gpu0.json
./h2d_driver_bench --device 1 --output h2d_results_gpu1.json

./h2d_driver_bench --device 0 --large-shards \
  --output results/h2d_large_shards_gpu0.json
./h2d_driver_bench --device 1 --large-shards \
  --output results/h2d_large_shards_gpu1.json

jq empty h2d_results_gpu0.json h2d_results_gpu1.json \
  h2d_results_summary.json
```

Python 基准使用本目录隔离环境：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-bench.txt

.venv/bin/python benchmarks/pinned_arena_probe.py \
  --output results/pinned_arena_probe.json

.venv/bin/python benchmarks/torch_llama32_1b_bench.py \
  --device 0 --warmup 5 --repetitions 30 --pipeline-stages 16 \
  --output results/torch_llama32_1b_gpu0.json

.venv/bin/python benchmarks/large_shard_pipeline_bench.py \
  --device 0 --group-layers 1,2,4,8 --rows 1,512,2048,4096 \
  --warmup 3 --repetitions 7 \
  --output results/large_shard_pipeline_gpu0.json

# 第二张卡反向测试顺序，用于检查 DVFS/温度/顺序偏差。
.venv/bin/python benchmarks/large_shard_pipeline_bench.py \
  --device 1 --group-layers 8,4,2,1 --rows 1,512,2048,4096 \
  --warmup 3 --repetitions 7 \
  --output results/large_shard_pipeline_gpu1.json

.venv/bin/python benchmarks/validate_large_shards.py
.venv/bin/python benchmarks/validate_results.py
.venv/bin/python benchmarks/build_report_artifact.py

node /path/to/data-analytics/skills/build-report/scripts/deliver_portable_artifact.mjs \
  --input artifact.json --output report.html
```

不传 `--output` 时 JSON 写到 stdout，进度写到 stderr。两张 GPU 应当串行
测试；并行测试会让两张卡争用 CPU 内存和 PCIe 路径，所得结果不是单卡
isolated baseline。

## 测量方法

每次传输用一对 CUDA Event 包围，并用
`cuEventElapsedTime` 取得 GPU timeline 时间。host latency 只包围一次
`cuMemcpyHtoD[_Async]` 函数调用：

- sync API 的 host latency 包含阻塞等待；
- pinned async 的 host latency 是 enqueue 返回时间；
- async GPU Event 时间包含实际 stream 中的 copy。

pageable 源用 4 KiB 对齐的普通 `posix_memalign`；pinned 源用
`cuMemHostAlloc`。两个 256 MiB host buffer 都先完整 `memset`，排除首次
缺页。各点先 warmup，再记录 10–200 个样本。百分位使用排序后的
`round(p * (n - 1))` 下标。GB/s 是十进制 GB/s。

必测尺寸为 0 B、1 B、4 KiB、64 KiB、1/4/16/64/116/256 MiB。为
Llama-3.2-1B 另加 2/8/20/32/96 MiB 和 121,643,008 B：

- K/V projection 各 2 MiB；
- Q/O projection 各 8 MiB；
- attention 权重合计 20 MiB；
- MLP 单矩阵 32 MiB，gate + up 64 MiB，MLP 合计 96 MiB；
- 矩阵合计 116 MiB；若把两个 4 KiB BF16 norm 算入，则整层为
  121,643,008 B。

“单次启动开销”另用 pinned async 批量测试：每批 1、2、4、…、1024
次 copy，对 batch median 拟合
`T_batch = intercept + copies * per_copy_slope`。0 B、1 B 和 4 KiB
分别拟合 host enqueue 与 GPU Event 总时间。

## 本机环境

- CPU：AMD Ryzen Threadripper 3970X，32 核/64 线程，单 NUMA node。
- GPU：2 × NVIDIA GeForce RTX 3080 Ti 12 GiB。
- GPU0 / GPU1：`0000:01:00.0` / `0000:48:00.0`，最大 PCIe Gen4 x16。
- 驱动：555.42.02；CUDA Driver API version 12050。
- Linux：5.15.0-139-generic x86_64。
- 两卡均报告 2 个 async engines，支持 concurrent kernels。
- `nvidia-smi topo -m` 显示 GPU0 与 GPU1 之间为 `SYS`。

系统 `ulimit -l` 只有 64 KiB，但本机 NVIDIA 驱动仍成功完成了
`cuMemHostAlloc(256 MiB)`；不要据此假设其他机器也一定能成功。

## 结果

以下都是 pinned `cuMemcpyHtoDAsync_v2` 的 median。完整 p10/p90 见原始
JSON。

| 分片 | GPU0 event | GPU0 GB/s | GPU1 event | GPU1 GB/s |
|---:|---:|---:|---:|---:|
| 4 KiB | 2.944 µs | 1.391 | 2.976 µs | 1.376 |
| 64 KiB | 5.344 µs | 12.263 | 5.376 µs | 12.190 |
| 1 MiB | 42.592 µs | 24.619 | 42.656 µs | 24.582 |
| 2 MiB | 82.816 µs | 25.323 | 82.720 µs | 25.352 |
| 8 MiB | 321.568 µs | 26.087 | 321.696 µs | 26.076 |
| 20 MiB | 799.328 µs | 26.236 | 799.744 µs | 26.223 |
| 32 MiB | 1.277 ms | 26.267 | 1.278 ms | 26.252 |
| 64 MiB | 2.703 ms | 24.832 | 2.797 ms | 23.992 |
| 96 MiB | 4.144 ms | 24.290 | 4.197 ms | 23.985 |
| 116 MiB | 5.018 ms | 24.242 | 5.055 ms | 24.062 |
| 116 MiB + 8 KiB | 5.020 ms | 24.232 | 5.055 ms | 24.066 |
| 256 MiB | 11.040 ms | 24.319 | 11.062 ms | 24.268 |

116 MiB pinned async 的稳定性：

| GPU | event p10 / median / p90 | GB/s p10 / median / p90 | host enqueue median |
|---:|---:|---:|---:|
| 0 | 5015.840 / 5017.568 / 5027.424 µs | 24.194 / 24.242 / 24.250 | 1.804 µs |
| 1 | 5050.528 / 5055.136 / 5059.168 µs | 24.042 / 24.062 / 24.084 | 1.824 µs |

大分片用 64/96/116/256 MiB 拟合 `T = alpha + size / B`：

| GPU / memory | alpha | B | R² |
|---|---:|---:|---:|
| GPU0 pinned async | -31.313 µs | 24.223 GB/s | 0.999936 |
| GPU1 pinned async | 59.761 µs | 24.390 GB/s | 0.999990 |
| GPU0 pageable async | 21.133 µs | 13.290 GB/s | 0.999961 |
| GPU1 pageable async | 111.357 µs | 13.381 GB/s | 0.999995 |

GPU0 的负 intercept 不是“负启动开销”；它说明 32–64 MiB 附近存在
cache/传输分段效应，单一直线不应被外推到小尺寸。规划大分片时使用约
24.0 GB/s 的保守值更可靠。

pageable async 在当前驱动上返回成功，但并不是真的 non-blocking：

| GPU | 116 MiB memory | host call median | event median | median GB/s |
|---:|---|---:|---:|---:|
| 0 | pageable async | 9098.613 µs | 9137.952 µs | 13.311 |
| 0 | pinned async | 1.804 µs | 5017.568 µs | 24.242 |
| 1 | pageable async | 9151.271 µs | 9184.096 µs | 13.244 |
| 1 | pinned async | 1.824 µs | 5055.136 µs | 24.062 |

因此调度器必须让权重驻留在真正 page-locked 的 CPU buffer 中。不能把
“pageable + Async 后缀”当作可重叠路径，而且这种接受 pageable 指针的
行为是驱动相关的。

批量启动拟合：

| GPU | copy size | host slope | GPU Event slope | GPU Event intercept | R² (GPU) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 B | 0.0587 µs | 0.0588 µs | 1.281 µs | 0.999926 |
| 1 | 0 B | 0.0571 µs | 0.0573 µs | 1.319 µs | 0.999866 |
| 0 | 1 B | 2.1304 µs | 2.1308 µs | -3.562 µs | 0.999925 |
| 1 | 1 B | 2.0992 µs | 2.0994 µs | -3.595 µs | 0.999924 |
| 0 | 4 KiB | 1.8341 µs | 1.8339 µs | 3.305 µs | 0.999971 |
| 1 | 4 KiB | 1.8203 µs | 1.8200 µs | 1.709 µs | 0.999925 |

0 B 被驱动优化成 no-op，其约 0.058 µs slope 只是参数检查/循环成本，
不是 DMA 启动成本。1 B 与 4 KiB 的约 1.82–2.13 µs slope 包含连续提交
和串行 tiny-copy command 成本；单独一次 pinned async 的实测 host call
median 约 1.8 µs。负 intercept 同样只是全区间最小二乘结果，不能赋予
物理含义。

## 对异步分层调度器的直接结论

1. CPU 权重必须预先进入 pinned pool。若模型最初由普通 pageable 内存
   载入，应由 CPU worker 提前拷入复用的 pinned staging slot，不能在
   compute 临界路径中临时锁页。
2. 使用至少两个 device weight slot：copy stream 把分片 `i+1` 搬入
   slot B 的同时，compute stream 用 slot A 执行分片 `i`。ready event
   从 copy stream 传给 compute stream；slot-reuse event 反向约束下一次
   覆盖。不要逐层 `cuStreamSynchronize`。
3. 64 KiB 只有约 12 GB/s，1 MiB 已达约 24.6 GB/s。建议调度粒度至少
   1–2 MiB；Llama 的 2 MiB K/V 矩阵已经足够接近饱和。更大的自然矩阵
   边界可减少 event/allocator/bookkeeping 数量。
4. 对 116 MiB 整层，下一层预取窗口约 5.0 ms；若当前层 GPU compute
   少于这个值，双缓冲仍会露出 H2D 尾巴。此时需要把调度粒度拆到
   projection/MLP 矩阵，并尽早发出后续分片，或用更深的 lookahead。
5. 双缓冲最少额外占用“两片权重 + activation/KV/cache/workspace”显存。
   实际 slot 大小应由显存预算和 `T_copy(chunk) ≈ T_compute(chunk)`
   共同决定，而不是固定为一整个 decoder layer。

本基准只测 isolated H2D。真实 overlap 还会受到 GPU DRAM 带宽竞争、
kernel occupancy、stream priority、功耗/时钟状态和 CPU staging worker
的影响；下一步必须用真实 Llama kernel 做“copy-only、compute-only、
overlap”三组 Nsight/CUDA Event 测量，并报告 overlap hidden fraction：

```text
hidden_fraction =
  (T_copy_only + T_compute_only - T_overlap) /
  min(T_copy_only, T_compute_only)
```

该值为 1 才表示较短的一侧被完全隐藏。
