# NVIDIA GPU 硬件兼容层

Cascade-LLM 根据 Compute Capability、运行时环境、Provider 能力和数值契约进行启动前检查。设备名称只用于展示，不参与兼容性判定。

当前正式边界：完整真实验证仅覆盖 SM86 RTX 3080 Ti。SM80、SM89、SM90 只有扩展骨架和声明，不能据此宣称 A100、RTX 4090 或 H100 已兼容。

## 状态语义

Provider 和报告统一使用 `declared`、`compiled`、`smoke_passed`、
`numerically_qualified`、`performance_qualified`、`production`、
`experimental`、`unsupported`。旧的 `unknown/unqualified/qualified/disabled`
不再是合法状态。

只有 `production` 表示 L0–L5 与长期性能门禁全部通过；
`numerically_qualified` 和 `performance_qualified` 必须按其字面范围解释。

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
