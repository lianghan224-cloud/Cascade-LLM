# KV 整改文档包使用说明

## 文件

- `KV_REMEDIATION_MASTER_PLAN.md`：唯一总规范，包含全部内容。
- `KV_CODEX_EXECUTION_PROMPT.md`：直接交给 Codex 的执行入口。
- `KV_VALIDATION_MATRIX.md`：验证部分的便捷摘录。
- `KV_ARCHITECTURE_CONTRACT_TEMPLATE.md`：Codex 审计后填写的契约模板。

## 复制到仓库

将这四个文件复制到：

```text
/disk2/home/guest/lianghan/repos/Cascade-LLM/docs
```

## 执行

```bash
cd /disk2/home/guest/lianghan/repos/Cascade-LLM
git status --short
codex exec "$(cat docs/KV_CODEX_EXECUTION_PROMPT.md)"
```

不需要新建分支。不要在执行前 reset、stash 或 clean。

## 完成后验证

```bash
python tools/validate_kv_stack.py --profile logic
```

并检查：

```text
docs/KV_REMEDIATION_IMPLEMENTATION_REPORT.md
reports/kv_validation.md
reports/kv_validation.json
```
