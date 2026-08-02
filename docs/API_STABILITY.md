# 核心接口稳定性策略

M5 起，以下接口进入 v1 冻结状态：

```text
ModelGeometry
WeightSpec
QuantizationSpec
CheckpointManifest
ExecutionPlan (schema v1)
LinearBackend
MixedRuntime
RunReport (schema v2)
```

冻结含义：

- dataclass 字段名及顺序不能原地修改；
- `ExecutionPlan` 与 `RunReport` 的不兼容序列化变更必须提升 schema version；
- `MixedRuntime.__init__`、`run()`、`close()` 的既有参数不能删除或改义；
- `LinearBackend` 的 `validate/transfer_bytes/workspace_bytes/execute` 协议不能原地改变；
- frozen backend 名称、storage/activation/output dtype 和 fallback 属性不能静默改变；
- 新字段优先放入新的 sidecar 配置或新 schema，不把旧字段改义。

当前机器可读契约在 `tests/fixtures/core_api_v1.json`，SHA-256 为：

```text
a699bcda404c8f3eeb9a4b297373dd3e8434c305f126acb4b3dc14a12ff70325
```

KV Framework 另有独立 V1 合同，冻结 Page Handle/Descriptor、HND layout、
Request/Batch/Slot Mapping、Store/Selection/Reuse 和 Paged Attention Provider ABI：

```text
fixture: tests/fixtures/kv_framework_v1.json
sha256: 0c8e71323062c7d21a18c1cd8703f549421b7b713abef6df796317f6010837dd
```

`tests/test_api_contract.py` 同时验证：

- frozen dataclass 字段快照；
- `WeightSpec`、`QuantizationSpec`、`CheckpointManifest` round-trip；
- `ExecutionPlan` JSON 确定性与未知版本拒绝；
- `RunReport` schema v2 round-trip 与未知版本拒绝；
- backend registry 与 `MixedRuntime` 方法签名。

`ExecutionPolicy.linear_backend` 是用户选择 sidecar，不属于 checkpoint
存储格式。它允许同一 INT8/INT4 checkpoint 明确选择 fallback 或注册的
fused provider，不修改 `WeightSpec` 和 `QuantizationSpec` 的冻结布局。

兼容变更流程：

1. 先增加读取旧版本的迁移代码。
2. 提升对应 schema version。
3. 保留旧 fixture，并增加新 fixture。
4. 增加旧 checkpoint/plan/report round-trip 测试。
5. 在开发文档中写明迁移和回滚方式。

禁止只更新 fixture 来绕过意外接口变化。任何 v1 hash 改变都必须先解释是
兼容修复还是新版本设计。
