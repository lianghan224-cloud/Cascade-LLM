# KV 验证矩阵执行摘录

> 本文件是 `KV_REMEDIATION_MASTER_PLAN.md` 第 8 节的执行摘录，不新增要求。冲突时以主计划为准。

## 状态语义

- `PASS`：测试真实执行并满足标准。
- `FAIL`：功能错误、未实现或结果不满足标准。
- `SKIPPED_WITH_REASON`：仅限缺 GPU、模型权重、真实 NVMe 或明确外部依赖。
- `BLOCKED`：被另一个已记录失败阻塞。

未实现功能不得标记为 SKIP。

## Logic Gate

无权重、无高端 GPU 条件下应完成：

- 生命周期 V02～V15。
- 共享和回滚 V16～V26。
- Quest 风格索引 V27～V38。
- KVDrive 风格 Mock 分层 V39～V51。
- 调度和 Provider 路由 V52～V59。

完整定义和通过标准见主计划第 8 节。

## 统一入口

```bash
python tools/validate_kv_stack.py --profile logic
python tools/validate_kv_stack.py --profile cuda-synthetic
python tools/validate_kv_stack.py --profile full
```

## 必须输出

```text
reports/kv_validation.json
reports/kv_validation.md
reports/kv_validation_failures/<case-id>/seed.txt
reports/kv_validation_failures/<case-id>/repro.json
```

失败必须返回非零退出码。
