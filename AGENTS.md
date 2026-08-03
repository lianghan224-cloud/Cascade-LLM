# Cascade-LLM Codex 工作约定

## 项目使命

Cascade-LLM 的长期目标是在单机、单 GPU 优先的条件下运行完整模型权重远大于 GPU 显存的 Llama-family 模型。完整权重保存在 CPU 内存或更低层存储中，GPU 只保留有界权重槽、必要常驻参数、激活和 KV；通过异步 H2D、计算重叠和显式内存预算换取可运行性。

正确性、显式失败和可复现证据优先于未经验证的性能声明。禁止用静默 fallback、隐藏的完整权重/KV materialization 或伪造 benchmark 掩盖未实现能力。

## 每次接手先做

1. 完整阅读 `docs/CODEX_HANDOFF.md` 和 `docs/SERVER_MIGRATION.md`。
2. 阅读 `README.md`、`docker/versions.env`、`requirements.lock`。
3. 涉及 KV 时再阅读 `docs/KV_ARCHITECTURE_CONTRACT.md` 和 `docs/KV_REMEDIATION_IMPLEMENTATION_REPORT.md`；不要把旧 KV 设计文档当成当前状态。
4. 运行 `git status --short`，保留用户已有修改，不清理、不 reset、不覆盖模型或结果文件。
5. 在新机器先运行 `tools/capture_environment.py` 和 `scripts/verify_server.sh`，再做模型实验。

## 当前优先级

- 当前主线：在更大 GPU/RAM 服务器上完成“大权重、权重远超显存”的真实实验。
- KV 管理暂时冻结在 **KV Stack Beta**；除非实验被 KV 正确性或显存直接阻塞，否则不要主动扩展 Quest、Tiered KV 或 Prefill Kernel。
- 第一个候选大模型下载入口是 `scripts/download_codellama34b_bf16.sh`，但 checkpoint 不属于 Git，且交接时尚未在本机完成真实 34B 资格验证。

## 环境与依赖规则

- 不复制 `.venv`、CUDA build cache、`__pycache__` 或本机编译的 `.so` 到另一台服务器。
- 推荐 Python 3.10；版本源是 `docker/versions.env`，Python 包锁定在 `requirements.lock`。
- 原生环境使用 `scripts/bootstrap_server.sh`；最严格复现使用 Docker，见 `docs/DOCKER_GUIDE.md`。
- 模型、HF token、OpenAI/Codex 凭据、SSH key、`.env` 和私有 prompt 永远不能提交。
- 新 GPU 必须按 Compute Capability 重新检查。SM86 是当前唯一完成生产级 KV 合成资格的架构；SM80/SM89/SM90 只有声明或有限验证，不能沿用 SM86 结论。
- 本机编译 Provider 必须在目标机重建；Generic CUDA paged kernel 由 NVRTC 在目标架构首次运行时编译。

## 架构边界

- 模型几何必须来自 checkpoint `config.json`，不得硬编码 8B/34B/70B shape。
- 启动顺序必须保持：metadata/checkpoint 校验 → MemoryPlanner 预检 → CPU arena → GPU 资源 → 推理。
- 权重格式、计算 backend、Provider、placement、KV policy 和 fallback 必须在报告中显式记录。
- `pinned_staging` 是大权重 bring-up 默认模式；只有 RAM、memlock 和实测收益足够时才使用 `full_pinned`。
- 大模型初测使用 batch 1、短 prompt、少量 decode token、紧凑 `max_cache_length`，先证明正确性和资源闭合，再扩大上下文。

## 验证命令

```bash
CASCADE_PYTHON_BIN=python3.10 bash scripts/bootstrap_server.sh
bash scripts/verify_server.sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q
.venv/bin/python tools/validate_kv_stack.py --profile logic
.venv/bin/python tools/validate_kv_stack.py --profile cuda-synthetic
```

`cuda-synthetic` 只能在实际 CUDA GPU 上运行。目标机不是 SM86 时，先使用 `generic_cuda` 建立正确性基线，再为目标架构单独资格验证。

## 实验记录要求

- 每次真实模型运行保存：Git commit、checkpoint/revision、GPU/driver、Python/Torch/CUDA、所有显式策略、峰值 RAM/pinned/GPU/KV、TTFT、decode ms/token、H2D/compute 分解和错误。
- 性能结论至少区分 Kernel microbenchmark 与真实端到端；权重流式主导时不得用 Kernel 加速比替代端到端加速比。
- 失败同样保存可复现命令和报告；不把 OOM、fallback 或缺少硬件改写成 PASS。
- 结果目录遵循 `.gitignore`：只提交紧凑、无敏感信息的摘要/manifest；原始 trace 和模型数据外部保存。

## 代码与 Git 约定

- 修改后运行与风险成比例的测试，并在交付中说明未运行项。
- 不提交本地绝对模型路径、凭据、编译二进制、虚拟环境或大模型文件。
- 不使用 `git reset --hard`、`git clean`、强推或改写历史，除非用户明确授权。
- 推送前检查 `git diff --check`、敏感信息、大文件和远端分支状态。
