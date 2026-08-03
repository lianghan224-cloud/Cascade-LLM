# KV Architecture Contract

## Status and compatibility

This document freezes the post-remediation KV semantics. The public KV Framework V1 dataclass and provider ABI remains unchanged and its frozen contract test still passes. Additional lifecycle metadata is attached to `PageDescriptor` without changing the frozen V1 dataclass field list; new Quest, tiering, scheduling and routing types are additive.

## Identity contract

| Identity | Code mapping | Rule |
|---|---|---|
| Logical KV block | `selection.LogicalKVBlockId` | Model, session, branch, layer and logical block are explicit and must not be substituted with a physical page ID. |
| Physical page handle | `PageHandle(page_id, generation, store_id, format_id)` | Every state/descriptor access and block-table compaction validates generation. |
| Index record | `IndexRecordId` and `QuestIndexRecord` | Record identity is stable across serialization and separate from a page pointer. |
| Location | `KVLocation` | Tier, device, slot, length, layout, dtype/quant format, version, state and checksum are explicit. |

The V1 `LogicalBlockTable` remains the request-to-physical-page map. The tiered location table maps stable logical block keys to one or more `KVLocation` values and one authoritative tier.

## Metadata contract

The semantic page/request/index/location state contains:

- request/session, branch and layer identity through request state and `LogicalKVBlockId`;
- token start/count through block position, request length and index record;
- capacity/layout/dtype/format through runtime, descriptor, store and location;
- `generation`, `data_version`, `index_version`;
- `ref_count`, `pin_count`, `inflight_compute`, `inflight_io`;
- page ownership state and per-tier residency state;
- locations, authoritative tier, reservations, dirty/error and checksum metadata.

`KVPagePoolV1` is the only production writer for generation, reference counts, pin counts and page-local version publication. `QuestCPUIndex` builds records, but `QuestFlatSelection` publishes them through `KVPagePoolV1.attach_index`.

## Ownership and residency states

The existing V1 page lifecycle remains compatible:

```text
FREE -> ALLOCATED -> ACTIVE -> SEALED/SHARED
                  -> COPYING -> ACTIVE
SEALED/SHARED -> MIGRATING/EVICT_PENDING
any releasable allocation -> RELEASING -> FREE
```

Tier copies use the orthogonal state model:

```text
ABSENT -> LOADING -> RESIDENT -> EVICTING -> ABSENT
                   \-> FAILED
```

A FAILED target never replaces the source authority.

## Required invariants

The following are runtime assertions and validation assertions:

1. `ref_count`, `pin_count`, `inflight_compute` and `inflight_io` never become negative.
2. `pin_count == inflight_compute + inflight_io` for physical pages.
3. A FREE page has zero refs, pins and inflight work, no logical mappings and appears exactly once in the free list.
4. A non-FREE page has at least one logical owner.
5. A stale or foreign `PageHandle` cannot expose state, descriptor data or a physical ID to a provider.
6. Shared-tail write and shared partial-tail rollback perform COW first.
7. A page cannot be finally released while pinned or busy.
8. A Quest record is queryable only when `index_version == data_version`.
9. Full Quest mode returns every legal block in logical order.
10. Each tiered logical block has exactly one resident authoritative tier at its committed version.
11. Migration may expose stable source replicas but never a half-written target.
12. Target copy/checksum/version failure preserves readable source authority.
13. Cancel and failure paths converge to zero reservations, IO work and transient pins.
14. Full/Chunked Prefill cannot route to a backend that advertises only Decode/Short Suffix.

## Migration commit protocol

`TieredKVStore.migrate` executes:

1. Validate source authority and target capacity.
2. Reserve target and increment source protection/inflight accounting.
3. Publish target as LOADING.
4. Read source bytes and write distinct target bytes/buffer/file.
5. Validate length, checksum and unchanged data version/authority.
6. Atomically mark target RESIDENT and switch the authoritative tier.
7. Optionally retain the source as a read-only replica or remove it inside the protected transaction.
8. On failure, delete/mark the target FAILED, restore source authority and unwind reservation/pin/IO counters.

## Quest index contract

`QuestCPUIndex` provides:

```text
build(block_data, metadata) -> QuestIndexRecord
update_append(record, appended_data, new_version) -> QuestIndexRecord
select(query, candidates, budget, mode, exact_scores=None) -> QuestSelectionResult
fork_ref(record) -> shared immutable record
cow_clone(record) -> isolated record
rollback(record, target_token_count, target_version) -> QuestIndexRecord
serialize(record) / deserialize(bytes)
validate(record, page_metadata)
```

The normal budget scorer reads per-dimension minimum/maximum summaries, not raw K/V rows. Rows are retained only for transactional append/rollback and `debug_exact` validation. Selection records candidate count, selected count, elapsed time, recall when exact scores are supplied and score error.

## Scheduler contract

`TieredAttentionCoordinator` accepts selector output or explicit logical IDs, admits requests FIFO, deduplicates prefetches through the store, waits with cancellation/timeout, pins every resolved block for compute, constructs an `AttentionKVView`, and always unpins on normal return or compute exception. Cancelled FIFO tickets are skipped so later requests cannot starve.

Pinned or inflight locations are excluded from public eviction. Capacity exhaustion either performs a legal LRU eviction with another valid authority or raises `KVCapacityError`.

## Workload routing contract

`PagedWorkload` has four explicit values:

- `full_prefill`
- `chunked_prefill`
- `decode`
- `short_suffix`

The current generic/SM86 CUDA attention backend advertises Decode and Short Suffix only. Full and Chunked Prefill route to `reference_paged_exact`, with `fallback_reason` recorded. This is correct but slower and remains an optimization target; it cannot silently call the decode-only CUDA mapping.

## Module responsibilities

- Runtime/Ownership/PagePool: allocation, release, refs, pins, generations, COW, rollback and final reclamation.
- Attention backend: compute on resolved selected views only.
- Quest index: index lifecycle, version validation and selection only.
- Tiered store: locations, copies, authority, reservations, prefetch and eviction only.
- Coordinator: composes selector, residency and compute pin lifetime.
- Dispatcher: workload classification, capability decision and explicit correctness fallback.

No backend may directly release a page or mutate request ownership counts.

## Error and diagnostics contract

Lifecycle errors identify page/generation or logical block/tier context. Quiesce is bounded and reports busy page IDs, generations, pins, inflight counts and states. Unified validation failures return nonzero, preserve seed `20260803` by default and write `reports/kv_validation_failures/<case-id>/seed.txt` plus `repro.json`.
