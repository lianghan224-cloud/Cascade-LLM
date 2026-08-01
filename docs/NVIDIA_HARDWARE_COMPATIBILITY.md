# NVIDIA GPU 硬件兼容层

Cascade-LLM 根据 Compute Capability、运行时环境、Provider 能力和数值契约进行启动前检查。设备名称只用于展示，不参与兼容性判定。

当前正式边界：完整真实验证仅覆盖 SM86 RTX 3080 Ti。SM80、SM89、SM90 只有扩展骨架和声明，不能据此宣称 A100、RTX 4090 或 H100 已兼容。

## 状态语义

Provider 使用 `unknown`、`declared`、`compiled`、`smoke_passed`、`qualified`、`production`、`unsupported`、`disabled`。其中只有 `qualified` 和 `production` 表示通过真实模型资格验证。

报告级 `unqualified` 不是 Provider 生命周期状态，而表示“请求可能可以执行，但没有达到 qualified”。例如 PyTorch fallback 已存在于当前构建中，但在一块新 GPU 上仍需完成硬件资格验证。

## 启动检查

显式 backend 请求会检查：

- Compute Capability 和已编译架构；
- Provider ABI 与扩展加载状态；
- prefill/decode phase；
- 权重格式、activation/scale dtype、group size；
- M/N/K 范围与对齐；
- physical layout 与 workspace 配置。

任一检查失败都会在执行计划运行前拒绝。fused Provider 不会自动切换到 fallback；prefill 和 decode 需要分别显式指定时，可使用：

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint MODEL \
  --provider cutlass \
  --prefill-backend int8_dequant_bf16_fallback \
  --decode-backend fused_w8a16
```

## 检查命令

```bash
.venv/bin/python tools/inspect_hardware.py --device cuda:0
```

```bash
.venv/bin/python tools/check_compatibility.py \
  --checkpoint MODEL \
  --backend fused_w8a16 \
  --phase decode \
  --m 1 \
  --load-cutlass
```

`RunReport` schema v2 未增加字段；硬件 profile、runtime feature 和兼容性结果写入既有 `hardware` 字典。
