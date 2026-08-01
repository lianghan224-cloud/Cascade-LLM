# Cascade-LLM 插件开发契约

Cascade 使用 Python entry point 发现三类插件：

```text
cascade_llm.providers
cascade_llm.model_adapters
cascade_llm.quantizers
```

入口实现位于 [plugins.py](../layer_streaming/plugins.py)。执行型命令采用 strict 加载：任一已安装插件加载失败都会拒绝运行；`doctor` 会保留错误并输出诊断报告。

## ProviderPlugin

Provider 插件必须暴露：

- 唯一名称；
- 完整 `ProviderCapability`；
- `create_provider()`；
- wheel 内精确 `.so` 路径；
- build metadata；
- 架构/ABI/layout 精确匹配的 Numerical Contract。

示例 entry point：

```toml
[project.entry-points."cascade_llm.providers"]
w8a16_sm86 = "cascade_provider.provider:create_plugin"
```

Bundle 安装器不会递归扫描共享库。它只安装 JSON 中明确列出的 qualified package，并在构建 wheel 前验证 ABI 和 compiled architecture。

## ModelAdapterPlugin

插件暴露 `model_types` 和 `create_adapter()`。加载后通过现有 `ModelAdapter` 协议注册，不修改 ExecutionPlan schema。

```toml
[project.entry-points."cascade_llm.model_adapters"]
qwen2 = "cascade_adapter_qwen2:create_plugin"
```

当前 core 只提供 Llama Adapter；Qwen2/Mistral 尚未实现或声明兼容。

## QuantizationPlugin

插件暴露唯一 `name` 和 `run(argv)`，负责完整转换、metadata、缓存 key 与错误回滚。Core 当前注册 `int8_per_channel`，内部复用现有 checkpoint 转换工具。

新增物理布局时，缓存 key 必须包含 Provider layout 和 ABI，禁止覆盖旧缓存。

## 资格边界

插件能够被发现不等于硬件兼容。Provider 必须依次满足：

```text
declared → compiled → smoke_passed → qualified → production
```

只有 `qualified` 或 `production` 项允许进入正式 Bundle；SM86 contract 不能用于 SM89，ABI/layout/dtype 变化也必须创建新 contract。
