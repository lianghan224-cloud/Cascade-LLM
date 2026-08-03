# Codex 一次性执行提示词

你正在 `/disk2/home/guest/lianghan/repos/Cascade-LLM` 当前工作树中执行 KV 子系统整改。

必须首先完整阅读：

```text
docs/KV_REMEDIATION_MASTER_PLAN.md
```

该文件是本任务唯一总规范，包含全部问题、任务、依赖、契约、Agent 分工、验证矩阵、暂缓规则和完成标准。不要只阅读摘要，也不要自行缩小范围。其他 KV 计划文件只是方便定位的摘录；冲突时以主计划为准。

执行要求：

1. 不新建或切换分支，不创建 worktree。
2. 不执行 reset、clean、stash、rebase、强制 checkout，不覆盖现有未提交修改。
3. 默认不 commit、不 push，直接在当前工作树修改。
4. 先保存 preflight status/diff/HEAD 快照，然后继续；工作树不干净不是中止理由。
5. 可以使用多个只读子 Agent 并行审计，但同一工作树中的代码写入必须按主计划的依赖顺序进行；公共文件只能由主控 Agent 修改。
6. 不要停留在审计、计划、空接口或 TODO。一次执行中尽量完成 T0～T10、运行验证、修复失败并生成最终报告。
7. 不依赖真实模型权重的功能必须实现和验证。缺 GPU、权重或真实 NVMe 的项目使用 SKIPPED_WITH_REASON；未实现功能不得标记 SKIP。
8. Quest 风格索引必须有 CPU reference、真实索引记录、full 模式、预算选择、版本一致性和统计。
9. KVDrive 风格分层必须有统一位置表、Mock GPU/CPU/SSD、真实数据复制、迁移状态机、唯一 authoritative version、预取去重、取消和失败回滚。
10. Runtime 必须统一 ref/pin/generation/version，并覆盖 Fork/COW/Prefix/Beam/Speculative/rollback。
11. Full Prefill 与 Decode 必须从 Provider/Dispatcher 路由层拆开；没有专用 Kernel 时使用正确 fallback，不得静默进入 Decode-only Kernel。
12. 创建或扩展 `tools/validate_kv_stack.py`，支持 `logic`、`cuda-synthetic`、`full`。
13. `logic` 必须覆盖主计划中所有无 CUDA/权重/NVMe 依赖项，并全部 PASS。
14. 失败必须返回非零退出码，保存 seed 和最小复现序列。
15. 禁止吞异常、降低断言、删除测试、扩大容差或只修改报告获得通过。
16. 遇到非破坏性的实现选择时自行采用最小合理方案并记录，不要反复向用户提问。
17. 最终必须生成：
    - `docs/KV_CURRENT_STATE_AUDIT.md`
    - `docs/KV_ARCHITECTURE_CONTRACT.md`
    - `docs/KV_REMEDIATION_IMPLEMENTATION_REPORT.md`
    - `reports/kv_validation.md`
    - `reports/kv_validation.json`
18. 最终回复必须逐项说明 T0～T10、Vxx 状态、修改文件、执行过的命令、未解决问题和硬件后续命令。

现在从 preflight 快照开始，持续执行到当前环境能完成的内容全部完成，不要在审计或中间阶段停止。
