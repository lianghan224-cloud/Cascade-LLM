# M5.1 Backend 验证记录

验证环境：

```text
Date:                 2026-07-29
GPU:                  NVIDIA GeForce RTX 3080 Ti
SM:                   86
PyTorch:              2.4.1+cu121
CUDA compiler:        12.1.105
CUTLASS:              v3.5.1
Host compiler:        GCC/G++ 9.4
```

## 完成项

### M5.1A BackendCapability 与资格测试

- `BackendCapability` 覆盖 M、SM、activation dtype、weight format、group
  size、K/N alignment。
- `tools/qualify_backend.py` 覆盖 tiny/8B/70B 的 hidden-hidden、up、down
  三类矩阵和 M=1/2/4/8/16/32/128/512。
- unsupported provider、格式、SM 和 shape 均在 kernel launch 前失败。

### M5.1B 显式 phase backend

- `BackendSelection` 和 `BackendPhasePlan` 作为 sidecar 存在。
- ExecutionPlan 仍为 schema v1，核心 API contract SHA256 未变化。
- RunReport v2 嵌套字段记录 prefill/decode backend、显式 decode fallback
  和实际 provider。

### M5.1C CUTLASS W8A16 SM86

- per-channel `M>=1`：CUTLASS Ampere mixed-input Tensor Core GEMM，
  M=1 已真实运行。
- per-group `M=1`：专用 weight-only CUDA GEMV。
- BF16/FP16 activation 和 BF16/FP16 scale 均通过。
- symmetric per-channel 支持 M>=1；symmetric per-group（group size
  32/64/128）支持 M=1 decode。
- 直接读取 INT8 weight，`workspace_bytes=0`，profile 中
  `dequant_ms=0`。
- provider 不创建 CUDA stream/event，使用当前 PyTorch compute stream。

per-group M>1 prefill、INT4、asymmetric 和其他 SM 会启动失败。per-group
checkpoint 可显式配置 fallback prefill + fused decode，不发生静默切换。

### M5.1E tiny fused E2E

以下路径通过：

```text
full_pinned + fused prefill + explicit fallback decode
pinned_staging + fused prefill + explicit fallback decode
pinned_staging + fused prefill + fused M=1 decode
pinned_staging + per-group fallback prefill + fused M=1 decode
resident vocab
streamed vocab benchmark
双 GPU slot / Copy Stream + Compute Stream
KV Cache prefill + decode
Hugging Face 显式解量化逐层/logits/Top-k 对比
```

重复调用 provider 500 次后，CUDA allocated/reserved 均无增长。

## 合成矩阵结果

8B metadata 生成的 24 个 BF16 case（三类矩阵 × 八个 M）全部通过当前
backend 数值门槛：

```text
maximum normalized relative error: 0.003113
minimum Top-k set consistency:      0.975
workspace:                          0
```

随机输出存在接近并列的 logit，因此部分 M>1 case 的 Top-1 顺序会变化；
该值已在资格 JSON 中单独记录，不能由合成矩阵推断真实模型生成一致性。
tiny 模型的完整 HF 对比和 fused/fallback Top-k 则保持一致。

8B metadata 的 per-group M=1 三类矩阵在 group size 32/64/128 下均通过。
BF16/FP16 activation 与 BF16/FP16 scale 四种组合已覆盖；最大归一化相对
误差为 `0.003748`，最低 Top-k 集合一致率为 `0.9`，Top-1 均为 `1.0`。
per-group M=2 会在静态资格检查中以 decode-only 原因拒绝。

8B M=1 单矩阵 CUDA event latency（同一工具、合成数据）：

| Matrix | fused ms | fallback ms | ratio |
|---|---:|---:|---:|
| hidden→hidden | 0.0799 | 0.3202 | 4.01x |
| hidden→intermediate | 0.0919 | 1.0722 | 11.66x |
| intermediate→hidden | 0.2627 | 1.0341 | 3.94x |

这些数字只证明 provider 相对 fallback 的单矩阵工程门槛，不是实际模型
tokens/s 结论。

70B 的三类 M=1 矩阵完成了真实 GPU 分配与数值运行；完整
M=1..512 capability metadata 共 24 case 通过且 workspace 均为 0。

## 未完成项

### M5.1D CUTLASS W4A16

未开始接入。对固定的 CUTLASS v3.5.1 源码完成了资格审计：

- SM80 mixed-input 单元覆盖 BF16/FP16 × INT8/UINT8，没有 stock
  BF16/FP16 × packed INT4 device GEMM；
- Ampere INT4 TensorOp 覆盖的是 INT4 × INT4，不能直接替代 W4A16；
- NVIDIA TensorRT-LLM 的 W4A16 使用额外的 `cutlass_extensions/fpA_intB_gemm`
  和 `weightOnlyBatchedGemv`，并区分 column-major/interleaved 预处理布局。

因此没有把一个逐元素 unpack kernel 注册成 `fused_w4a16`。下一步必须先
冻结 provider 物理布局与 checkpoint 转换/缓存契约，再移植经过验证的
weight-only extension；不能复用 W8A16 名称或把 INT4 fallback 冒充 fused。

参考：

- https://github.com/NVIDIA/cutlass/blob/v3.5.1/test/unit/gemm/warp/gemm_mixed_input_sm80.cu
- https://github.com/NVIDIA/cutlass/blob/v3.5.1/test/unit/gemm/device/gemm_s4t_s4n_s4t_tensor_op_s32_sm80.cu
- https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv
- https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm

### M5.1F 本地真实小 checkpoint

在 `/disk2/home/guest/lianghan` 范围内未找到 `model.safetensors` 或
`model.safetensors.index.json`，因此真实 Llama-family 验收尚未完成。
当前通过的 checkpoint 均由合成生成器创建，不能冒充真实模型兼容证明。

本地 checkpoint 可用后执行：

```bash
.venv/bin/python tools/accept_checkpoint.py \
  --checkpoint /path/to/local-llama \
  --output /tmp/cascade-real-acceptance.json \
  --backend checkpoint \
  --weight-store pinned_staging \
  --vocab-mode streamed
```

若 checkpoint 是 Cascade 支持的 symmetric INT8 per-channel 格式，再追加
`--provider cutlass --backend fused_w8a16` 验收。per-group checkpoint
使用显式 fallback prefill + fused decode。
