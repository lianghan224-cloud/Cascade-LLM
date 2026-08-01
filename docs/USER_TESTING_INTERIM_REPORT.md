# Cascade-LLM 中间用户测试报告

**测试日期：** 2026-08-01
**测试版本：** M6 第一轮
**测试范围：** 单机、单 GPU、单请求基础推理路径
**总体结论：** 基础运行验收全部通过，框架核心显存调度和 W8A16 加速效果符合预期；CLI 易用性仍需改进。

## 1. 测试环境

* 操作系统：Linux
* GPU：2 × NVIDIA GeForce RTX 3080 Ti
* 单卡显存：12 GiB
* Compute Capability：SM86
* PyTorch：2.4.1+cu121
* CUDA Runtime：12.1
* Python：3.8.10
* 测试设备：`cuda:0`
* BF16 模型：Llama-3.1-8B-Instruct
* W8A16 模型：Llama-3.1-8B-Instruct-W8A16

该环境符合当前 CUTLASS W8A16 provider 的 SM86 要求。测试依据为基础用户测试手册，暂不覆盖服务端、batch、多 GPU 和正式长基准。

## 2. 测试结果汇总

| 测试项                     | 结果   | 主要结果                             |
| ----------------------- | ---- | -------------------------------- |
| GPU、CUDA、Python 环境检查    | 通过   | CUDA 可用，SM86，模块导入正常              |
| fused provider 检查       | 通过   | `fused_w8a16` 可用，支持 SM86         |
| BF16 checkpoint 校验      | 通过   | 无 missing、shape、dtype 或 shard 错误 |
| W8A16 checkpoint 校验     | 通过   | 元数据校验成功                          |
| 基础 W8A16 冒烟测试           | 通过   | 无 fallback，Transformer 100% 常驻   |
| UT-01 BF16 最低显存路径       | 通过   | BF16 全流式可在 12 GiB GPU 上运行        |
| UT-02 W8A16 全流式路径       | 通过   | H2D 约为 BF16 的一半                  |
| UT-03 显存—速度曲线           | 通过   | 显存预算增加时速度持续提升                    |
| UT-04 LM Head placement | 通过   | LM Head 常驻明显降低 decode 延迟         |
| UT-05 显存不足预检            | 功能通过 | 在生成前正确拒绝，但错误输出不够友好               |

## 3. 主要性能结果

### 3.1 推荐 W8A16 常驻模式

配置为 Transformer 和 LM Head 常驻时：

* backend：`fused_w8a16`
* provider：`cutlass_sm86_w8a16`
* fallback：无
* Transformer resident hit ratio：`1.0`
* Transformer H2D：`0`
* GPU peak：约 `7.72 GiB`
* Decode：约 `45.4～45.7 ms/token`
* Decode 吞吐：约 `21.9 token/s`
* 生成文本正常

实际运行报告确认 checkpoint、内存预检、后端和 provider 均符合预期。

### 3.2 BF16 与 W8A16 全流式对比

| 路径        |       Decode 延迟 | Transformer H2D/forward |   GPU peak |
| --------- | --------------: | ----------------------: | ---------: |
| BF16 全流式  | 约 1452 ms/token |              约 13.0 GiB | 约 0.45 GiB |
| W8A16 全流式 |  约 785 ms/token |               约 6.5 GiB | 约 0.26 GiB |

W8A16 全流式相对 BF16 全流式快约 `1.85×`，权重传输量约下降一半。BF16 全流式测试中 resident hit ratio 为 0，且未发生 fallback。

### 3.3 显存预算与速度曲线

| Transformer 常驻预算 | Resident hit ratio | Transformer H2D |          Decode |
| ---------------: | -----------------: | --------------: | --------------: |
|            0 GiB |             0.0000 |       6.500 GiB | 794.52 ms/token |
|            4 GiB |             0.5938 |       2.641 GiB | 416.33 ms/token |
|            8 GiB |             1.0000 |           0 GiB | 136.01 ms/token |

结果表明：

* 显存预算增加时，常驻命中率单调上升；
* Transformer H2D 单调下降；
* Decode 延迟持续下降；
* 三组测试 CV 均低于 5%，短基准稳定性合格。

### 3.4 LM Head 常驻效果

| LM Head 模式 | LM Head H2D |  GPU peak |          Decode |
| ---------- | ----------: | --------: | --------------: |
| streamed   |   0.979 GiB | 6.734 GiB | 136.01 ms/token |
| resident   |       0 GiB | 7.720 GiB |  45.41 ms/token |

LM Head 常驻增加约 `0.99 GiB` 显存，但 Decode 延迟降低约 66.6%，速度提升约 `3×`。

## 4. 已发现问题

### 4.1 CLI 显存错误输出不友好

显存不足预检能够正确拒绝运行：

```text
estimated GPU peak is 15.90 GiB, but only 11.39 GiB is free
```

但当前直接输出完整 Python traceback。建议捕获 `MemoryPreflightError`，只显示简洁错误和处理建议。

### 4.2 Provider 字段表现不一致

`check_backends.py` 的 provider 字段显示 Python 模块路径，而生成报告中的 provider 为 `cutlass_sm86_w8a16`。建议统一字段语义或分别命名为：

* `provider_module`
* `provider_name`

### 4.3 weight_format 字段容易误解

W8A16 checkpoint 的 `weight_format` 显示为 `int8_dequant_bf16_fallback`，该值更接近 fallback backend 名称。建议将 checkpoint 格式与执行 backend 分开记录。

### 4.4 Memlock 依赖仍需说明

测试机 `RLIMIT_MEMLOCK` 约为 64 MiB，低于 pinned staging 需求，因此测试使用了 `--ignore-memlock-limit`。该参数只跳过预检，不能保证所有机器都能成功分配 pinned memory。

### 4.5 SSH 连接异常需复核

UT-05 完成后 SSH 连接关闭。当前无法确定是网络问题、终端行为还是进程影响，建议单独复测，不应直接判定为框架缺陷。

## 5. 阶段结论

Cascade-LLM 当前已经证明以下核心能力：

1. BF16 8B 模型可以通过全流式方式在单张 12 GiB GPU 上运行。
2. W8A16 fused 路径可以正常执行，且没有静默 fallback。
3. 权重常驻预算能够有效控制显存占用与推理速度。
4. Transformer 和 LM Head 完全常驻后，Decode 可达到约 `45 ms/token`。
5. 显存不足能够在实际生成前被预检发现。
6. JSON 报告能够反映 backend、placement、H2D、显存和延迟等核心指标。

当前核心运行能力通过验收，但用户入口仍偏向开发者工具。下一阶段建议补充 CLI 易用性测试，包括 `--help`、非法参数、模型路径错误、简化运行参数、Ctrl+C 退出和错误信息可读性。

## 6. P0 数值问题处理结果

fused 与 fallback 的 5 个失败 stage 已完成逐项定位。结论不是 placement、scale layout、alignment 或 padding 错误，而是 PyTorch BF16 `F.linear` 的 reduced-precision reduction/cuBLAS 归约顺序与 fused M=1 FP32 累加 GEMV 的舍入语义不同；微小局部差异经 32 层传播后使旧 stage allclose 门槛失效。

已新增：

- 448 linear/两步 decode 的逐矩阵同输入诊断；
- 三个固定真实 prompt、共 1344 linear 的 golden evidence；
- versioned golden contract；
- local、FP32 accumulation、stage 和 Top-k 四层回归门禁；
- 一键真实 checkpoint golden runner。

三组 golden 全部通过 contract v1，六次 logits Top-1 全一致，Top-10 集合交集最低 90%。详细根因、5 个 stage、Layer 31 矩阵范围和正式门槛见 `docs/FUSED_W8A16_NUMERICAL_INVESTIGATION.md`。旧的失败项仍保留在原始报告中，没有删除或通过修改 tolerance 将其伪装为 allclose 通过。
