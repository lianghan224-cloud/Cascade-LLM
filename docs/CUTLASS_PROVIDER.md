# CUTLASS SM86 W8A16 Provider

## 支持边界

当前 backend 名称为 `fused_w8a16`，实际 provider 标识为
`cutlass_sm86_w8a16`。第一版只声明：

```text
GPU:                 SM86
activation/output:   BF16 或 FP16
weight:              symmetric INT8 per-channel；
                     symmetric INT8 per-group（仅 M=1）
scale:               BF16 或 FP16
quantization axis:   1（linear input/K 轴）
group size:          32 / 64 / 128
K alignment:         16
N alignment:         8
M:                   per-channel >= 1；per-group = 1
workspace:           0
```

per-channel 的 `M>1` 使用 CUTLASS Ampere mixed-input Tensor Core GEMM，
GEMM 后由同一 stream 上的 CUDA kernel 应用 per-channel scale。M=1 decode
和 per-group M=1 使用专用 weight-only CUDA GEMV，直接读取对应 channel/group
的 scale，并先在 activation dtype 中完成权重解量化舍入，再以 FP32 累加。
这样可避免 decode 在 INT8 GEMM 输出舍入后才应用 scale 的额外误差。两条路径
都直接读取 INT8 weight，不创建完整 BF16/FP16 weight workspace。

per-group 的 M>1、asymmetric INT8、INT4、SM86 以外 GPU 和未满足
alignment 的矩阵会在启动阶段被明确拒绝，不会静默进入 fallback。
per-group prefill 后续仍需接入 fine-grained scale mainloop。

当前固定版本的 stock SM80 mixed-input 模板不直接提供 BF16/FP16 × packed
INT4 device GEMM；W4A16 需要额外的 weight-only extension 和物理布局契约，
审计结论见 `docs/M5_1_VALIDATION.md`。因此 `fused_w4a16` 仍保持 unavailable。

## 构建

provider 是独立 C ABI 共享库，不依赖 Python/PyTorch C++ 头文件。Python
只通过 `ctypes` 传入 CUDA tensor 指针和当前 PyTorch stream。

已验证组合：

```text
CUTLASS v3.5.1
CUDA compiler 12.1.105
GCC/G++ 9.4
PyTorch 2.4.1+cu121
RTX 3080 Ti / SM86
```

构建命令：

```bash
CUDA_HOME=/path/to/cuda-12.1 \
CUTLASS_ROOT=/path/to/cutlass-3.5.1 \
.venv/bin/python -m layer_streaming.providers.cutlass.build
```

也可完全显式指定：

```bash
.venv/bin/python -m layer_streaming.providers.cutlass.build \
  --cutlass-root /path/to/cutlass-3.5.1 \
  --nvcc /path/to/cuda/bin/nvcc \
  --cuda-runtime-root /path/to/cuda-runtime
```

默认输出
`layer_streaming/providers/cutlass/_build/libcascade_cutlass_sm86.so`。
构建产物被 `.gitignore` 排除，不提交本机二进制。

## 能力检查和资格测试

```bash
.venv/bin/python tools/check_backends.py \
  --device cuda:0 --provider cutlass --require fused_w8a16

.venv/bin/python tools/qualify_backend.py \
  --backend fused_w8a16 --provider cutlass \
  --preset 8b --matrix all \
  --m 1 2 4 8 16 32 128 512 \
  --activation-dtype bf16 --scale-dtype bf16 \
  --output /tmp/cascade-cutlass-8b.json
```

资格报告包含 capability、unsupported reason、workspace、alignment、SM、
数值误差、Top-k 一致率和 CUDA/wall latency。工具使用固定随机种子，并标记
`synthetic_only=true`，不能作为真实模型性能结论。

per-group decode 资格测试：

```bash
.venv/bin/python tools/qualify_backend.py \
  --backend fused_w8a16 --provider cutlass \
  --preset 8b --matrix all --m 1 \
  --granularity per_group --group-size 32
```

资格 JSON 始终记录 Top-1 与 Top-k。门禁使用 backend 平均/相对误差和
Top-k 集合一致率（当前最低 90%）；Top-1 只记录不作为随机矩阵门禁，因为
近似并列输出会在合法舍入误差内改变次序。

真实 8B decode 的 fused/fallback 舍入差异、逐 linear 同输入诊断、三个
固定 prompt 的 golden contract 和一键回归命令见
`docs/FUSED_W8A16_NUMERICAL_INVESTIGATION.md`。不得用修改通用 allclose
阈值代替该版本化门禁。

## 显式阶段计划

prefill/decode 都使用 fused：

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint /path/to/int8-per-channel-checkpoint \
  --provider cutlass \
  --prefill-backend fused_w8a16 \
  --decode-backend fused_w8a16
```

显式 prefill fused、decode fallback：

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint /path/to/int8-per-channel-checkpoint \
  --provider cutlass \
  --prefill-backend fused_w8a16 \
  --decode-backend int8_dequant_bf16_fallback
```

per-group 当前必须显式使用 fallback prefill、fused decode：

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint /path/to/int8-per-group-checkpoint \
  --provider cutlass \
  --prefill-backend int8_dequant_bf16_fallback \
  --decode-backend fused_w8a16
```

阶段选择保存在 `BackendPhasePlan` sidecar 中，不修改 ExecutionPlan schema
v1。若任一阶段需要的 workspace 超过冻结 plan 的预算，启动阶段直接失败。
RunReport v2 的嵌套字段记录 phase backend、显式 fallback 和实际 provider。

## 第三方依赖

CUTLASS 是 header-only 构建依赖，固定验证版本为 NVIDIA CUTLASS v3.5.1：

- https://github.com/NVIDIA/cutlass/tree/v3.5.1
- https://github.com/NVIDIA/cutlass/blob/v3.5.1/LICENSE.txt

CUTLASS 使用 BSD-3-Clause。分发编译后的 provider 二进制时必须同时保留
其版权声明、许可条件和免责声明；完整文本见 `docs/THIRD_PARTY_NOTICES.md`。
