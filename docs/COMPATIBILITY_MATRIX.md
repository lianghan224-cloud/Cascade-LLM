# NVIDIA 兼容矩阵

更新时间：2026-08-01。该表区分“框架声明”和“真实资格验证”。设备名称仅为示例，实际以 Compute Capability 为准。

| 架构 | 基础 Torch 路径 | CUTLASS W8A16 | 当前结论 |
| --- | --- | --- | --- |
| SM75 | compiled，未完成本项目硬件资格验证；BF16 不可用 | unsupported | unqualified |
| SM80 | compiled，未完成 A100 真实资格验证 | declared，未编译 | unqualified |
| SM86 | compiled；RTX 3080 Ti 已完成现有回归 | qualified，ABI 2，RTX 3080 Ti | qualified（仅已测组合） |
| SM89 | compiled，未完成真实资格验证 | declared，未编译 | unqualified |
| SM90 | compiled，未完成真实资格验证 | declared，未编译 | unqualified |
| unknown | 不选择 | 不选择 | unsupported |

`compiled` 的 Torch 路径表示代码随当前 PyTorch 构建存在，不代表所有列出的 GPU 已完成 Cascade-LLM 资格验证。

机器可读矩阵可通过以下命令生成：

```bash
.venv/bin/python tools/export_compatibility_report.py \
  --output compatibility_matrix.json
```

当前项目可以声明已预留 SM80、SM89、SM90 兼容框架，但不得声明已支持 RTX 4090、A100、H100 或全部 CUDA GPU。
