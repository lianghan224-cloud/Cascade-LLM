# Provider 构建指南

构建框架支持单架构请求和 fat-binary 架构列表，但当前只有 SM86 W8A16 有实际 kernel 实现。其他架构只能生成 `declared` 元数据，不能生成或标记为 `compiled`。

## SM86 实际构建

```bash
.venv/bin/python tools/build_provider.py \
  --architecture sm86 \
  --provider w8a16 \
  --cutlass-root /path/to/cutlass \
  --nvcc /path/to/nvcc \
  --cuda-runtime-root /path/to/cuda-runtime
```

成功后生成共享库和相邻的 `.metadata.json`。元数据记录 ABI、实际编译架构、格式和构建环境；无法可靠读取的版本写为 `unverified`。

Provider 二进制只记录 `libcudart.so.12` 依赖，不写入构建主机的绝对 CUDA `RPATH/RUNPATH`。容器运行时由 NVIDIA CUDA Runtime 镜像的标准动态链接配置解析依赖；旧的带主机绝对路径二进制必须重新构建后才能作为正式容器 Provider 发布。

## 未实现架构的构建声明

```bash
.venv/bin/python tools/build_provider.py \
  --architectures sm80,sm86,sm89,sm90 \
  --provider w8a16 \
  --metadata-only \
  --output build/provider_metadata/w8a16-plan.json
```

这个文件的状态是 `declared`，`compiled_architectures` 为空。去掉 `--metadata-only` 会被拒绝，避免产生虚假的 fat binary 能力表。

## 新架构接入规则

每个新架构必须有独立 Provider 名称、ABI、编译目标、物理布局和 Numerical Contract。不得通过扩大 SM86 Provider 的 `supported_architectures` 来宣称 SM89 支持。
