# Cascade-LLM Docker 使用指南

Docker 交付固定使用一个多阶段 [Dockerfile](../docker/Dockerfile)、一个正式 [Compose](../compose.yaml) 和统一 `cascade` 命令。模型永远不进入镜像。

当前状态：D1～D4 的容器、CLI、插件 Bundle、Compose 和一键脚本骨架已经落地；GHCR 正式镜像尚未发布。Generic 镜像可从源码构建。Full/SM86 镜像要求构建上下文中存在已经完成资格验证的 SM86 ABI 2 `.so`，缺失时构建会明确失败。

## 宿主机要求

- Linux；
- NVIDIA Driver；
- Docker Engine 和 Compose v2；
- NVIDIA Container Toolkit；
- 本地 Llama-family safetensors checkpoint。

Docker 不能修复过旧驱动、缺失 Toolkit、模型授权、CPU RAM/磁盘不足或未验证 GPU 架构。

## 快速开始

推荐通过脚本调用。脚本自动读取唯一版本源 `docker/versions.env`：

```bash
export CASCADE_MODEL_DIR=/ssd/cascade-llm/models
export CASCADE_CACHE_DIR=/ssd/cascade-llm/docker-cache
export CASCADE_RESULT_DIR=$PWD/results

./scripts/cascade-docker.sh install
./scripts/cascade-docker.sh build
./scripts/cascade-docker.sh doctor
```

运行 BF16 checkpoint：

```bash
./scripts/cascade-docker.sh run \
  --checkpoint /models/Llama-3.1-8B-Instruct \
  --backend bf16_linear \
  --max-new-tokens 32
```

也可以直接调用 Compose，但构建时必须显式加载集中版本文件：

```bash
docker compose --env-file docker/versions.env build
docker compose --env-file docker/versions.env run --rm cascade doctor
```

## 镜像类型

- `generic`：BF16/FP16 和显式 INT8/INT4 fallback，不安装 fused Provider。
- `qualified`：只安装 `qualified/production` Bundle 项；当前只有 SM86 W8A16 ABI 2。
- `sm86`：当前与 qualified Provider 集合相同，但标签明确架构。
- `devel`：包含 nvcc、编译器、测试和 Provider 开发环境。

构建 Generic：

```bash
CASCADE_PROVIDER_BUNDLE=generic \
CASCADE_IMAGE_TYPE=runtime \
./scripts/cascade-docker.sh build
```

构建 SM86（需先在固定工具链下生成本地 `.so`）：

```bash
CASCADE_PROVIDER_BUNDLE=sm86 \
CASCADE_IMAGE_TYPE=sm86 \
./scripts/cascade-docker.sh build
```

SM80、SM89、SM90 Bundle 只有 declaration，没有可安装 Provider；它们不会自动启用 fused backend。

## 固定挂载

| 容器目录 | 权限 | 用途 |
|---|---:|---|
| `/models` | 只读 | checkpoint/tokenizer |
| `/config` | 只读 | 用户配置 |
| `/cache/huggingface` | 可写持久化 | Hugging Face 缓存 |
| `/cache/cascade` | 可写持久化 | 量化和 Provider 缓存 |
| `/results` | 可写持久化 | RunReport、benchmark、资格报告 |
| `/opt/cascade/providers` | 镜像内只读 | Provider 插件源/构建清单 |
| `/opt/cascade/adapters` | 镜像内只读 | 外部 Adapter 预留目录 |

正式容器以非 root 用户运行、根文件系统只读，并启用 `no-new-privileges`。只将 `/tmp`、cache 和 results 设置为可写。

## 稳定 CLI

长期冻结以下入口：

```text
cascade doctor
cascade inspect
cascade validate
cascade run
cascade chat
cascade benchmark
cascade qualify
cascade quantize
cascade shell
```

`validate` 会先检查显式 backend 的 prefill/decode 兼容性，再执行 checkpoint、tokenizer 和 MemoryPlanner 验证。`run`、`validate` 和 `benchmark` 自动把报告写到 `/results`。

`cutlass_w8a16` 是 CLI 友好别名，内部仍解析成冻结 backend 名称 `fused_w8a16`。

`cascade quantize --input /models/MODEL` 未指定输出时，会根据 checkpoint 内容、geometry、量化配置、layout、Provider ABI 和转换器版本生成缓存 key，并写入 `/cache/cascade/quantized/sha256-...`。已有完整 manifest 时直接复用；显式 `--output` 则保持原工具语义。

## Full 镜像构建失败

以下错误是预期安全门禁：

```text
qualified provider binary is missing
provider ABI mismatch
provider compiled architecture mismatch
```

不要把 Bundle 改为 `declared` 来绕过。只有真实二进制、build metadata 和 Numerical Contract 三者匹配时，Provider wheel 才会安装。
