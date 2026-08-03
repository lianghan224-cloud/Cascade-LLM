# KV Current State Audit

## Audit scope and evidence

This audit was performed against the current worktree on 2026-08-03 before and during remediation. It covers `layer_streaming/kv`, `layer_streaming/attention/paged`, architecture providers, KV tests and KV tools. The searchable inventories are:

- `reports/kv_remediation/kv_symbol_inventory.txt`
- `reports/kv_remediation/kv_test_inventory.txt`
- `reports/kv_remediation/preflight_git_status.txt`
- `reports/kv_remediation/preflight_worktree.patch`
- `reports/kv_remediation/preflight_index.patch`
- `reports/kv_remediation/preflight_untracked_files.txt`
- `reports/kv_remediation/preflight_head.txt`

The preflight snapshot showed no tracked worktree or index diff. Existing untracked plans, reports, result data, scripts and downloads were preserved. No branch, worktree, stash, reset, clean, rebase, commit or push was used.

## System classification

Before this remediation, the production path was accurately classified as a paged KV ownership runtime with partial paged-attention kernels. Quest and CPU/NVMe store types were declarations only: `QuestFlatSelection`, `HierarchicalQuestSelection`, `PinnedCPUKVStore` and `NVMeKVStore` advertised `implemented=False` or inherited unsupported methods.

After this remediation, the current milestone is **KV Stack Beta** with:

- generation-safe paged ownership and failure assertions;
- Fork/COW/Prefix/Beam/Speculative/rollback coverage;
- a real Quest-style CPU summary index and sparse runtime adapter;
- a real Mock GPU/CPU/SSD location table and migration state machine;
- selection-to-prefetch-to-compute scheduling;
- explicit Full/Chunked Prefill, Decode and Short Suffix routing;
- logic and SM86 synthetic CUDA validation.

It is not Production Qualified because real model, dataset, long-context service and real NVMe/GDS validation are not configured.

## State ownership map

| State or concept | Canonical writer | Readers / consumers | Audit conclusion |
|---|---|---|---|
| `generation` | `KVPagePoolV1.allocate` | `KVPagePoolV1.descriptor`, `PageHandle.state`, block-table compaction | Old handles are rejected before physical IDs reach a provider. |
| `ref_count` | `KVPagePoolV1.allocate/retain/release` | `OwnershipManager`, invariants, reports | No production backend writes ownership counts. The only other writer is the isolated `experimental/legacy_kv_cache.py`, excluded from the production import graph. |
| `pin_count`, compute/IO inflight | `KVPagePoolV1.pin/unpin/begin_copy/end_copy` | release checks, quiesce, reports | Compute and IO accounting are distinguished and must sum to total pins. |
| Page lifecycle state | `KVPagePoolV1` through allocation, activation, seal, copy and release | Ownership manager and reports | FREE invariants and busy-state release rejection are executable assertions. |
| Request length/version | `OwnershipManager.commit/reset/rollback/commit_branch` | batch builder, selector, attention | Append, partial commit and rollback update length, page table and versions together. |
| Logical block table | `OwnershipManager` via `LogicalBlockTable` | batch builder and selector | Contains generation-bearing `PageHandle`, never raw pointers. |
| Prefix ownership | `PrefixCache` | prefix index and runtime | Cache-held and request-held references are distinct; cache eviction cannot free an active request's page. |
| Quest index records | `QuestCPUIndex`, published through `KVPagePoolV1.attach_index` | `QuestFlatSelection` | Query requires `index_version == data_version`; stale records fail. |
| Tier locations and authority | `TieredKVStore` | coordinator and diagnostics | One authoritative tier is committed only after copy/checksum/version validation. |
| Provider route | `PagedAttentionDispatcher` | attention execution and reports | Full/Chunked Prefill cannot enter the current Decode/Short-Suffix CUDA mapping. |

## Production call chains

### Append and commit

```text
PagedKVRuntime.append
  -> KVExecutionCoordinator.append
  -> OwnershipManager.begin_append
       -> allocate and/or tail COW through KVPagePoolV1
  -> PagedKVKernelBackend.append_kv
  -> QuestFlatSelection.build_request_layer (only when selected)
  -> OwnershipManager.commit
       -> sequence/page/version atomic commit
```

Kernel backends receive page views and slot mappings. They do not allocate, retain, release or change request tables.

### Attention

```text
PagedKVRuntime.attend
  -> build_paged_batch_view
       -> PageHandle generation check before raw page-ID compaction
  -> selection.select
  -> PagedAttentionDispatcher workload classification
  -> OwnershipManager compute pins
  -> selected attention backend or correctness fallback
  -> CUDA Event record
  -> bounded drain/quiesce and unpin
```

### Fork, COW and rollback

```text
fork -> seal + retain each shared page -> immutable Quest record sharing
append to shared tail -> allocate + copy + replace -> release old branch reference
rollback -> preflight releasability -> optional shared-tail COW -> truncate/release
         -> length/version update -> Quest tail rebuild and stale-record removal
```

### Tiering and scheduling

```text
Quest/full selection
  -> TieredAttentionCoordinator FIFO admission
  -> residency check
  -> deduplicated TieredKVStore.prefetch
  -> source authority pin + target reservation
  -> real byte copy to buffer/file + checksum
  -> atomic authoritative-tier commit
  -> compute pin + attention view
  -> paired unpin, including compute failure
```

## Release entry points

- `PagedKVRuntime.release` delegates to `OwnershipManager.release`.
- `PagedKVRuntime.reset` delegates to `OwnershipManager.reset`.
- `discard_branch` delegates to runtime release.
- `commit_branch` releases the parent's prior table after checking pins/busy states.
- `PrefixCache.evict_all/close` releases only cache-owned references.
- Append and COW exception paths release newly allocated targets.
- Rollback releases only truncated suffix pages.
- Runtime close quiesces, releases requests, releases prefix ownership, closes store data and marks the runtime closed.

All physical page releases converge on `KVPagePoolV1.release`.

## Provider and store boundaries

- `PagedAttentionBackend`: compute only; no ownership fields or release calls.
- `PagedKVKernelBackend`: append/copy bytes only; no ownership authority.
- `QuestCPUIndex`: build/update/select/fork/COW/rollback/serialize/validate; no page release or tier migration.
- `TieredKVStore`: locations, reservations, migration, prefetch and eviction; no request `ref_count` changes.
- `TieredAttentionCoordinator`: composes selection, residency and compute pins; does not own data versions.

## Tests found and gaps closed

Existing tests already covered stable arenas, reference numerics, generation reuse, Fork/COW, Prefix, provider contracts, SM86 synthetic kernels and lifecycle cycles. The baseline is preserved in `reports/kv_remediation/baseline.*`.

The remediation added executable coverage for:

- 100,002 seeded randomized lifecycle operations and 2,000 concurrent shared-ref/allocation cycles;
- double release/unpin, OOM, kernel and IO failure injection;
- partial/cross-page rollback and prefix eviction with an active owner;
- Quest full/budget/append/Fork/COW/rollback/stale/serialization/boundaries;
- real Mock-tier byte movement, checksum, atomic authority, failure rollback, cancel and dedup;
- selection-to-prefetch-to-view, compute failure unpin and FIFO fairness;
- route-specific Full/Chunked Prefill fallback and Decode/Short Suffix selection;
- CUDA Decode/Prefill numerics, two streams, allocator drift and failure recovery.

The final all-test regression ran 152 tests successfully. The complete V00-V76 evidence is in `reports/kv_validation.md` and `reports/kv_validation.json`.

## Remaining environment-dependent gaps

- Real checkpoint logits/Top-1, 8B long generation and real long-context Prefill.
- Dataset-backed Quest accuracy.
- Dedicated real NVMe Direct IO/GDS throughput and IO/compute overlap.
- Non-SM86 architecture-specific tuning.
- Large service-style dynamic batching with real weights.

These are environmental follow-ups, not hidden substitutes for missing logic implementations.
