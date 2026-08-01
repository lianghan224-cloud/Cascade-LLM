# NVIDIA 硬件资格验证指南

资格验证按固定六级顺序执行，工具不会自动提升 Provider 状态。

1. 环境：设备识别、Compute Capability、CUDA、显存、Pinned Memory、Stream/Event。
2. 基础 backend：BF16、FP16、INT8/INT4 显式 fallback。
3. Fused 单算子：M=1 与 prefill M、真实 N/K、dtype、group 和 alignment 边界。
4. Tiny 模型：prefill、decode、KV、resident/streamed 和 mixed placement。
5. 真实模型：checkpoint、逐层结果、logits、Top-k、长 decode 和显存稳定性。
6. 性能与长期稳定：1000 token、P50/P95/P99、内存和 stream/event 漂移。

完成 Level 5 才能标记 `qualified`；完成 Level 6 并纳入持续回归后才能标记 `production`。

## 生成资格工作单

```bash
.venv/bin/python tools/qualify_hardware.py \
  --device cuda:0 \
  --output qualification/environment.json
```

输出只把 Level 1 标记为 `observed`，总体状态保持 `unqualified`。后续各级必须附带原始 JSON、日志、模型哈希、Provider ABI 和 Numerical Contract key，由维护者审查后更新注册状态。

## Numerical Contract

Contract key 包含 architecture、Provider 名称与 ABI、模型 geometry、weight/activation/scale dtype 和 physical layout。查找只允许完全匹配：SM86 golden 不能用于 SM89，ABI 或 layout 变化也必须建立新 contract。
