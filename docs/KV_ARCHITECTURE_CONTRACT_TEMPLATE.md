# KV 架构契约模板

> 本模板用于 Codex 根据真实仓库代码生成 `docs/KV_ARCHITECTURE_CONTRACT.md`。完整要求位于 `KV_REMEDIATION_MASTER_PLAN.md` 第 6 节。

## 1. 真实类型映射

| 契约语义 | 仓库类型/文件 | 备注 |
|---|---|---|
| LogicalKVBlockId | 待审计 | 逻辑身份，不等于物理页 |
| PhysicalPageHandle | 待审计 | 必须包含 generation |
| KVLocation | 待审计 | tier/device/offset/layout/version |
| IndexRecord | 待审计 | 必须有 index_version |
| KVPageMetadata | 待审计 | 生命周期、位置和版本统一语义 |

## 2. 唯一写入者

| 状态 | 唯一写入模块 | 允许读取模块 |
|---|---|---|
| ref_count | KV Runtime | Backend/Index/Tiered Store |
| pin_count | KV Runtime | Backend/Tiered Store |
| generation | KV Runtime | 所有页面访问方 |
| data_version | KV Runtime/受控写接口 | Index/Tiered Store/Backend |
| index_version | Index Backend | Runtime/Scheduler |
| authoritative_location | Tiered Store 受控提交 | Runtime/Scheduler/Backend |

## 3. 不变量

```text
ref_count >= 0
pin_count >= 0
FREE => ref=0, pin=0, no inflight, no logical mapping
stale generation handle => rejected
shared page write => COW first
queryable index => index_version == data_version
one logical block => one authoritative version
pinned page => not final-free or destructive-evict
failure/cancel/rollback => converges to legal state
```

## 4. 迁移协议

```text
reserve target
protect source
copy
validate
atomically commit authoritative location/version
release or retain source replica
rollback target on failure
```

## 5. Provider 路由

必须分别记录 Full Prefill、Chunked Prefill、Decode、Short Suffix 的 capability、实现和 fallback。
