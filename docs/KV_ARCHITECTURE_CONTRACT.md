# KV Architecture Contract

## Status and compatibility

This document freezes the post-remediation KV semantics. The public KV Framework V1 dataclass and provider ABI remains unchanged and its frozen contract test still passes. `PagedBatchViewV2` is the additive, versioned batch ABI for generation metadata; V1 callers retain the original field list and constructor. New RGKV, tiering, scheduling and routing types are additive. Historical Quest types remain compatibility adapters only.

New system-level reports use exactly this capability vocabulary:

```text
NOT_IMPLEMENTED
EXPERIMENTAL
LOGIC_VALIDATED
CUDA_SMOKE
QUALIFICATION_READY
QUALIFIED
```

`QUALIFICATION_READY` means implementation, fault injection and report runners
are ready to execute; it is never equivalent to `QUALIFIED`. Frozen Provider
ABI labels such as `numerically_qualified` remain readable for compatibility
but are conservatively mapped by `layer_streaming.capability_state`; Provider
numerics alone map no higher than `CUDA_SMOKE` at the system level.

## Identity contract

| Identity | Code mapping | Rule |
|---|---|---|
| Logical KV block | `selection.LogicalKVBlockId` | Model, session, branch, layer and logical block are explicit and must not be substituted with a physical page ID. |
| Physical page handle | V1 fields plus additive `pool_uuid` | Every pool lookup validates Pool UUID, descriptor identity, generation, store and format; equal page IDs in different runtimes are never interchangeable. |
| Index record | `RGKVPageSummary` (`QuestIndexRecord` compatibility adapter) | Record identity is stable across serialization and separate from a page pointer. |
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

`KVPagePoolV1` is the only production writer for generation, reference counts, pin counts and page-local version publication. `RGKVIndex` builds summaries, but `RGKVSelectionPolicy` publishes them through `KVPagePoolV1.attach_index`.

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
4. Every reference has one explicit logical owner and `ref_count == logical_owner_count` for every page, including request, branch, Prefix Cache and temporary allocation owners.
5. A stale or foreign `PageHandle` cannot expose state, descriptor data or a physical ID to a provider.
6. Shared-tail write and shared partial-tail rollback perform COW first.
7. A page cannot be finally released while pinned or busy.
8. An RGKV summary is queryable only when `index_version == data_version`.
9. Dense/Prefill selection returns every legal block in logical order.
10. Each tiered logical block has exactly one resident authoritative tier at its committed version.
11. Migration may expose stable source replicas but never a half-written target.
12. Target copy/checksum/version failure preserves readable source authority.
13. Cancel and failure paths converge to zero reservations, IO work and transient pins.
14. Full/Chunked Prefill cannot route to a backend that advertises only Decode/Short Suffix.
15. Source and target of COW remain IO-pinned in `COPYING` until their `KVOperationFence` completes; metadata publication happens after the Fence.
16. Page `data_version` comes from one runtime-global strictly increasing epoch and is independent of request-local `RequestKVState.version`.

## Operation fence contract

`KVOperationFence` is the additive asynchronous operation record. It carries
operation/request identity, kind, source/target handles, CUDA Event or IO
Future, status/error/cancel state and submit/completion epochs. Append abort
first drains every submitted layer fence. COW failure either reaches bounded
quiescence and rolls metadata back, or preserves the busy pins/state and
reports cleanup diagnostics without hiding the original failure.

Attention follows the same contract. `SelectedPageView` first resolves the
logical selection back to generation-bearing handles and rejects logical,
physical or generation mismatch. Ownership then de-duplicates and pins only
the physical pages that the selected Provider view can read. A pending
Attention Fence therefore implies a positive page pin/inflight-compute count;
`wait_attention_fence`, `drain_attention_fence`, request release, reset and
runtime close are explicit drain points. A Provider exception records a failed
Fence and reaches bounded quiescence before releasing temporary pins.

Mock tier Prefetch and Migration also expose `KVOperationFence`. A
`RequestScopedPrefetchGroup` owns the consumer relationship for one request and
provides bounded wait, cancel, quiesce and cleanup. A timeout cancels remaining
work, waits for the worker to leave its critical section, and only then allows
reservation/pin cleanup. De-duplicated copies may have multiple request
consumers; cancelling one consumer cannot cancel work still required by
another.

## Prefix capacity contract

Each cached prefix is an explicit `PrefixEntry` with namespace/hash, handles,
page and byte accounting, creation/access epochs and hit count. Only full,
sealed pages are admissible. `max_prefix_pages` and `max_prefix_bytes` are
independent optional limits; budget enforcement evicts the least recently used
entry. Physical accounting is based on unique generation-bearing handles so
chained entries do not double-charge or double-retain shared pages. Replacing
the same hash publishes the new entry and releases the old Prefix owner exactly
once. Eviction releases only the Prefix owner, never a live Request owner.

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

The active tensor path is additive to this mock fault model:

- `GPUHotKVCache` owns only bounded physical slot/location metadata; a
  `PageHandle` remains a logical Ownership identity and is never a slot ID.
- One hot slot covers every layer of a Token Page Bundle. Pinned CPU
  reservations remain Layer Page allocations and publish atomically only
  after every required layer replica is valid.
- `ActiveTierCoordinator` performs real tensor D2H/H2D, deduplicated
  request-scoped Prefetch, LRU eviction and compute-wave fencing. It never
  retains or releases a Request reference.
- Eviction excludes PagePool/local pins, compute/IO inflight work, pending
  Append/COW, active operations and the current attention wave.
- CUDA slot reuse occurs only after the wave's compute-stream Event has
  quiesced and both PagePool and hot-slot pins have been released.
- Data Epoch is revalidated before every migration or append publication;
  failure/cancel keeps the valid source authority and removes an unpublished
  target.

When Dense exact Selection exceeds the hot capacity,
`TieredStreamingExactAttention` loads a bounded page wave and executes a
two-pass softmax: global maximum first, then denominator/numerator. Full KV
may not be staged in a hidden GPU tensor. The current executable scope is
batch 1 and request-only. Dense reads the complete logical context; RGKV reads
only its selected logical set and Active Tier may prefetch only selected,
non-GPU-resident pages. Unsupported Active Tier fork/COW/Prefix/rollback and
NVMe combinations fail explicitly.

## RGKV index contract

The canonical `RGKVIndex` / `RGKVSelectionPolicy` contract provides:

```text
build(block_data, metadata) -> RGKVPageSummary
update_append(record, appended_data, new_version) -> RGKVPageSummary
select(query, candidates, budget, mode, exact_scores=None) -> RGKVSelectionResult
fork_ref(record) -> shared immutable record
release_ref(record) -> delete at zero adapter owners
cow_clone(record) -> isolated record
rollback(record, target_token_count, target_version) -> RGKVPageSummary
serialize(record) / deserialize(bytes)
validate(record, page_metadata)
```

The normal budget scorer reads per-dimension minimum/maximum/mean summaries,
not raw K/V rows. Runtime records use contiguous tensors plus token/Data Epoch
metadata and retain no per-token Python rows. Append rebuilds only changed
logical pages; rollback rebuilds only the affected tail; sealed historical
summaries remain unchanged. The sparse budget is strict:
`total_page_budget = mandatory_recent_pages + relevance_selected_pages`, then
logical order is restored for the kernel. It never adds Recent pages after
Top-k and never truncates relevant old pages by logical ID.

`torch_tensorized` is an explicit opt-in scorer. It packs the frozen compact
records on the query device and performs score, stable rank/top-k, recent-page
merge and final logical ordering without `.cpu()`, `.tolist()` or `.item()` in
the Decode selection path. The CPU compact index remains the ownership and
quality oracle. Full/Chunked Prefill always returns an exact all-page view
because index publication follows the cross-layer append commit; an
uncommitted index is never exposed early. Policy, Runtime, CLI and
MemoryPlanner must record the scorer and separately budget CPU index, GPU
resident index and temporary selection workspace. This logic validation does
not qualify RGKV quality or end-to-end performance, so RGKV remains
opt-in/Experimental.

`DeviceKVPageTable` is an execution metadata mirror, never a lifecycle owner.
It carries an `int32` physical GPU hot slot (or `-1`), valid tokens,
authoritative Data Epoch, Page generation and location flags. Tensorized RGKV
returns the additive `DeviceSelectedPageView` ABI with contiguous device
tensors for logical/PagePool IDs, GPU slots, valid tokens, CSR boundaries,
expected/current epoch, expected/current generation, location, validity and
error masks. PagePool physical IDs and GPU hot-slot IDs are different
namespaces and may never be substituted for one another.

Candidate and selected epoch, generation and valid-token mismatches are
rejected as `STALE_INDEX` through device-visible asynchronous assertions.
`DevicePagedAttentionInput` and `Dispatcher.execute_device` form a strict
Decode-only Provider ABI: an unsupported or Reference backend raises
`UNSUPPORTED_DEVICE_SELECTED_VIEW` and must not convert to a Host view or
silently fallback. Generic CUDA consumes `gpu_physical_slots` directly.

The all-GPU-resident Active Tier route must acquire an ownership-neutral
`HotCacheExecutionGuard` before Location metadata gather. Slot assign/remap,
release and eviction remain frozen until its compute-stream Event Fence
completes. The guard changes no Request ref, Page owner or PagePool pin. This
closes the zero-readback slot-reuse race conservatively at arena granularity.
Host `SelectedPageView` resolution remains permitted only for Reference/Debug
and the not-yet-device-driven Tier-miss path.

## Generation session contract

`GenerationSession` is the batch-one lifecycle adapter over the model executor
and KV runtime. Its states are `NEW`, `PREFILLED`, `DECODING`, `FINISHED`,
`CANCELLED` and `CLOSED`. It owns Prefill, continuation, one-token Decode,
streaming, configured sampling, cancel, reset and idempotent close. Text is
encoded/decoded only through the supplied tokenizer; BOS/EOS and special-token
semantics are never hard-coded. `prefill_messages()` must call the tokenizer's
HF chat template and explicitly fails when no template exists. EOS, stop-token,
cross-token stop-string, maximum-length and cancellation termination share one
finish-reason contract. Streaming returns token ID, safe text delta, terminal
flag and finish reason; possible stop-string prefixes are held until they are
safe to publish. Continuation submits only the pending generated token plus the
new suffix, preserving existing Prefix KV, and rejects a rendered chat prefix
mismatch rather than recomputing silently. Cancel/reset/close quiesce KV Fences
before clearing or releasing request resources. A lifecycle lock covers each
complete Prefill/Decode model step and its state publication, so concurrent
cancel/reset/close cannot release a Request between Transformer layers.
Cleanup failures are attached
to, but never mask, the original model/Provider exception. This API is not a
scheduler and does not implement Continuous Batching.

## Scheduler contract

`TieredAttentionCoordinator` accepts selector output or explicit logical IDs, admits requests FIFO, deduplicates prefetches through the store, waits with cancellation/timeout, pins every resolved block for compute, constructs an `AttentionKVView`, and always unpins on normal return or compute exception. Cancelled FIFO tickets are skipped so later requests cannot starve.

Pinned or inflight locations are excluded from public eviction. Capacity exhaustion either performs a legal LRU eviction with another valid authority or raises `KVCapacityError`.

## Workload routing contract

`PagedWorkload` has four explicit values:

- `full_prefill`
- `chunked_prefill`
- `decode`
- `short_suffix`

The current generic/SM86 CUDA attention backend advertises Decode and Short Suffix only. Full and Chunked Prefill route to the explicitly configured Prefill provider, with cumulative workload/provider/fallback counts recorded. The default remains `reference_paged_exact`. An opt-in experimental `gather_sdpa_prefill` path gathers complete per-layer KV, accounts that workspace in MemoryPlanner, and invokes SDPA; it is not the default until real 70B qualification. Neither path may silently call the decode-only CUDA mapping.

Every Prefill provider also exposes additive `validate_shape(shape)` and
`estimate_workspace_shape(shape)` preflight methods. MemoryPlanner resolves
the configured Provider without loading CUDA code and consumes this estimate;
it no longer duplicates Provider-specific gather/workspace formulas. A
Provider estimate must cover every explicit gather/expand/mask/output tensor
plus its declared internal safety allowance. Any measured CUDA peak above the
Provider estimate is a strict qualification failure; the global CUDA safety
margin cannot be used to hide a Provider-specific underestimate.

## Resource and qualification evidence contract

`KVResourceSnapshot` is the common before/after record for Page, Owner, Pin,
Inflight work, pending Fence/Append, Prefix, RGKV, CUDA allocated/reserved,
CPU RSS, pinned bytes, thread and Future counts. `cuda_reserved` growth with
stable `cuda_allocated` is allocator cache, not automatically a leak; strict
checks may capture again after `torch.cuda.empty_cache()`. Unavailable pinned
or child-process metrics carry an explicit source marker and are never
invented as measured zero.

All new qualification runners emit `environment.json`, `cases.json`,
`summary.json` and `report.md`. Qualification mode performs a selected physical
GPU process check before execution and the 70B runner repeats that check around
each case. Qualification additionally requires scheduler-backed GPU allocation
evidence; a point-in-time process snapshot alone is not a reservation.
External compute work produces `BLOCKED_NOT_EXCLUSIVE`; an explicit shared run
is always `SMOKE_ONLY`. Logic and CUDA smoke modes cannot invoke the 70B model.

## Module responsibilities

- Runtime/Ownership/PagePool: allocation, release, refs, pins, generations, COW, rollback and final reclamation.
- Attention backend: compute on resolved selected views only.
- RGKV index: index lifecycle, version validation and selection only.
- Tiered store: locations, copies, authority, reservations, prefetch and eviction only.
- Coordinator: composes selector, residency and compute pin lifetime.
- Dispatcher: workload classification, capability decision and explicit correctness fallback.

No backend may directly release a page or mutate request ownership counts.

## Error and diagnostics contract

Lifecycle errors identify page/generation or logical block/tier context. Quiesce is bounded and reports busy page IDs, generations, pins, inflight counts and states. Unified validation failures return nonzero, preserve seed `20260803` by default and write `reports/kv_validation_failures/<case-id>/seed.txt` plus `repro.json`.
