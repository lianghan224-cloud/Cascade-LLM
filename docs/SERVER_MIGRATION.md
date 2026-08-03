# Cascade-LLM 同局域网服务器迁移手册

本手册的目标是：使用 Git 搬源码、使用锁文件重建依赖、使用独立数据通道搬模型和结果，并让目标服务器上的 Codex 能从仓库内说明继续工作。

## 1. 不要直接复制整个项目目录

当前源码与受控文档只有几十 MiB，本机 `.venv` 约 5 GiB。直接 rsync 整个项目会同时搬走：

- 绑定旧 Python 解释器路径的虚拟环境；
- 针对旧 GPU 架构生成的 NVRTC/CUDA cache 和 `.so`；
- `__pycache__`、临时报告和可能包含私有 prompt 的运行状态；
- 与新机器 driver/glibc 不兼容的二进制。

正确分三条通道：

1. 源码和紧凑证据：GitHub。
2. checkpoint/HF cache/大结果：局域网 rsync 或在目标机重新下载。
3. Codex、GitHub、HF 凭据：在目标机单独登录，绝不进入仓库。

## 2. 目标服务器基线

推荐基线与当前锁文件保持一致：

| 项目 | 推荐值 |
|---|---|
| OS | Ubuntu 22.04 x86_64 |
| Python | 3.10 |
| PyTorch | 2.4.1 |
| Torch CUDA runtime | 12.1 |
| CUDA image | 12.1.1 |
| Transformers | 4.45.2 |
| NVIDIA Driver | 能运行 CUDA 12.1，且先用 `nvidia-smi` 验证 |
| 磁盘 | 仓库、checkpoint、HF cache、结果各自独立预算 |
| RAM | 大于 checkpoint arena、staging、系统余量之和 |

版本源：`docker/versions.env`；Python 包：`requirements.lock`。

目标 GPU 不必与旧机器相同，但必须记录 Compute Capability：

```bash
nvidia-smi
python3 -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
```

SM86 是当前唯一完成完整 KV 合成资格的架构。A100/SM80、Ada/SM89、H100/SM90 迁移后先走 `generic_cuda` 正确性基线，再单独资格验证。

## 3. 从 GitHub 拉取源码

在目标服务器选择明确的数据盘路径，例如：

```bash
mkdir -p /srv/cascade
cd /srv/cascade
git clone --branch framework-v0 https://github.com/lianghan224-cloud/Cascade-LLM.git
cd Cascade-LLM
git rev-parse HEAD
git status --short
```

第一次继续开发前确认：

```bash
git remote -v
git branch --show-current
git log -3 --oneline
```

不要把旧服务器的 `.git` 和新 clone 混合，也不要在没有确认远端分支时直接覆盖目标目录。

## 4. 搬迁模型和大结果

模型不在 GitHub。局域网优先使用可断点续传的 rsync：

```bash
# 在旧服务器执行；替换目标用户和主机。
rsync -aH --partial --info=progress2 \
  /disk4/llm_model/ \
  target-user@target-host:/srv/cascade/models/
```

如果只搬一个 checkpoint：

```bash
rsync -aH --partial --info=progress2 \
  /path/to/Model-Name/ \
  target-user@target-host:/srv/cascade/models/Model-Name/
```

对极重要的 checkpoint，在两端生成校验文件：

```bash
cd /path/to/Model-Name
find . -type f -print0 | sort -z | xargs -0 sha256sum > /tmp/model.sha256
sha256sum -c /tmp/model.sha256
```

也可以在目标机使用项目下载脚本重新获取固定 revision：

```bash
bash scripts/download_codellama34b_bf16.sh --dry-run
# 确认许可证、revision、目标目录和空间后再正式运行。
```

HF token 只通过目标机环境或官方登录流程提供，不写入仓库、`.env.local`、shell history 或交接文档。

## 5. 重建 Python 环境

### 方案 A：原生虚拟环境，适合开发和 Codex

先安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10 python3.10-venv python3.10-dev \
  build-essential binutils ninja-build git ca-certificates
```

然后在仓库根目录运行：

```bash
CASCADE_PYTHON_BIN=python3.10 bash scripts/bootstrap_server.sh
```

脚本会：

1. 拒绝非 Python 3.10，避免悄悄产生另一套环境；
2. 创建新的 `.venv`，不会删除已有环境；
3. 从 `docker/versions.env` 读取工具链版本；
4. 安装 `requirements.lock`；
5. 以 editable、no-deps 方式安装当前仓库；
6. 执行 `pip check`、导入检查和环境快照。

不要复制旧 `.venv`。如果目标路径已经存在不可信 `.venv`，先人工移动到隔离目录，再运行脚本。

### 方案 B：Docker，适合最严格依赖复现

宿主机需安装 Docker、Compose v2 和 NVIDIA Container Toolkit：

```bash
export CASCADE_MODEL_DIR=/srv/cascade/models
export CASCADE_CACHE_DIR=/srv/cascade/cache
export CASCADE_RESULT_DIR=/srv/cascade/results

bash scripts/cascade-docker.sh install
bash scripts/cascade-docker.sh build
bash scripts/cascade-docker.sh doctor
```

默认先构建 `generic`。不要把旧服务器生成的 SM86 `.so` 复制进新架构镜像。详细说明见 `docs/DOCKER_GUIDE.md`。

## 6. 目标机验收

原生环境建立后运行：

```bash
bash scripts/verify_server.sh
```

它在本地忽略目录 `reports/migration/` 写入：

- `environment.json`：OS、Git、Python、依赖、Torch/CUDA、GPU、锁文件 hash；
- `doctor.json`：快速 CUDA/运行时检查；
- `hardware.json`：项目兼容性报告；
- `unittest.log`：仓库单元测试输出。

之后按架构运行 KV 验证：

```bash
.venv/bin/python tools/validate_kv_stack.py --profile logic
.venv/bin/python tools/validate_kv_stack.py --profile cuda-synthetic
```

注意：KV validator 会更新 `reports/kv_validation.*`。如果只是目标机本地验收，不要把这两个机器特定结果直接提交；先审阅差异和硬件含义。

验收标准：

- `git status` 中没有无法解释的源码变化；
- `pip check` 为 0；
- doctor/inspect 能识别 GPU、driver 和 Compute Capability；
- 单元测试通过；
- Generic CUDA 可建立新架构正确性基线；
- 目标机编译产物来自目标机，不来自旧机器；
- 模型路径可读，但不位于 Git 工作树内。

## 7. Provider 与 CUDA 迁移规则

- `generic_cuda` 使用 NVRTC，第一次运行会按目标 GPU 编译；首次延迟不是稳态 benchmark。
- `layer_streaming/providers/*/_build/*.so` 被 Git 忽略，必须在目标机重建。
- SM86 attention/KV bundle 只用于 Compute Capability 8.6。
- SM86 CUTLASS W8A16 binary、metadata、ABI、Numerical Contract 必须一起匹配；不得改状态字段绕过门禁。
- 新架构先运行 `cascade inspect` 和 `tools/check_compatibility.py`，再选择 Provider。
- CUDA OOM 排查先看 MemoryPlanner 中 weight slots、resident vocab、KV `max_cache_length` 和 activation，不要先修改 allocator。

## 8. 在目标服务器配置 Codex

以下步骤依据 OpenAI 当前官方 Codex CLI、认证和 `AGENTS.md` 文档。

### 安装或更新

macOS/Linux 官方安装器：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex --version
```

安装和更新使用同一命令。官方入口：

- https://developers.openai.com/codex/cli/
- https://developers.openai.com/codex/auth/

### 登录

有浏览器回调时：

```bash
codex login
codex login status
```

远程/headless 服务器优先设备码登录：

```bash
codex login --device-auth
```

API key 模式适合计量付费或自动化：

```bash
printenv OPENAI_API_KEY | codex login --with-api-key
```

不要把 `~/.codex/auth.json` 复制进仓库。它包含访问凭据；如确实需要在受信服务器之间复制，只能走单独的加密/SSH 通道并设置严格文件权限。

### 项目工作说明

Codex 启动时会读取仓库根目录的 `AGENTS.md`。在仓库根目录执行：

```bash
codex
```

第一条消息使用 `docs/CODEX_HANDOFF.md` 第 7 节的建议指令。也可以先核验加载的说明：

```bash
codex --ask-for-approval never "列出并总结当前仓库加载的 instruction 文件，不做修改。"
```

个人默认配置位于 `~/.codex/config.toml`；项目级 `.codex/config.toml` 只在信任仓库后加载。建议个人配置从保守权限开始：

```toml
approval_policy = "on-request"
sandbox_mode = "workspace-write"
```

不要把个人 model entitlement、token、代理地址或私有 MCP 凭据写入项目配置。

可选添加官方 OpenAI Developer Docs MCP：

```bash
codex mcp add openaiDeveloperDocs --url https://developers.openai.com/mcp
codex mcp list
```

修改 MCP 或配置后重新启动 Codex 会话。

## 9. GitHub 和 SSH 凭据

推荐在目标服务器单独配置 SSH key 或 GitHub CLI：

```bash
ssh -T git@github.com
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL"
```

如将 remote 切换到 SSH：

```bash
git remote set-url origin git@github.com:lianghan224-cloud/Cascade-LLM.git
```

SSH private key、GitHub token 和 credential helper 数据不能进入仓库或结果报告。

## 10. 第一个大权重实验

不要直接跑长上下文。顺序是：

1. capture environment；
2. checkpoint manifest/config 验证；
3. MemoryPlanner metadata/preflight；
4. batch 1、短 prompt、1 token decode；
5. 保存完整 RunReport；
6. 检查资源释放；
7. 才增加 token、context、slots 和 resident budget。

大权重默认策略见 `docs/CODEX_HANDOFF.md`。当前 Full/Chunked Prefill 会走 correctness fallback，因此不能用它评价最终 TTFT 性能。

## 11. 常见迁移错误

| 症状 | 首先检查 |
|---|---|
| `torch.cuda.is_available() == False` | driver、容器 Toolkit、设备权限、Torch CUDA wheel |
| `invalid device function` | 复制了旧架构 binary/cache；删除本地 build cache 后目标机重编译 |
| Provider architecture mismatch | Compute Capability 与 bundle 名称不一致 |
| import/ABI 错误 | 是否绕过 `requirements.lock`、是否复制旧 `.venv` |
| pinned allocation 失败 | RAM、`ulimit -l`、容器 memlock、使用 `pinned_staging` |
| 启动即 OOM | max cache、resident vocab/weights、slot 数、activation 和安全余量 |
| 长 Prompt 极慢 | 当前 Full/Chunked Prefill correctness fallback，不是权重流式结论 |
| clone 后 Codex 不理解项目 | 是否从 Git 根目录启动、是否读取 `AGENTS.md` 和 handoff |
