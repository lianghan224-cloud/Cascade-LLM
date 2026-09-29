# Cascade-LLM MoE Architecture

## Scope and qualification boundary

This implementation is additive. `WeightSpec`, `ModelGeometry` and
`ExecutionPlan` schema v1 remain unchanged, the Dense executor never performs
a `WeightObjectCatalog` lookup, and the KV runtime was not modified.

Current qualification status is **NOT_QUALIFIED**. The code and synthetic/CUDA
evidence cover T1-T12 and T14, but the required real
`allenai/OLMoE-1B-7B-0924` weight shards were unavailable after repeated
download failures. Consequently neither `MOE_FUNCTIONAL` nor
`MOE_LOW_VRAM_READY` is claimed.

## Execution structure

```text
Dense (unchanged)                         MoE (additive sidecars)
ExecutionPlan v1                          WeightObjectCatalog
  -> WeightSpec                             -> OlmoeModelAdapter
  -> MultiDtypeWeightStore                  -> resident Router
  -> LinearBackend                          -> ExpertDispatcher
                                                -> ExpertExecutionPlan
                                                -> ExpertScheduler
                                                   -> ExpertResidencyManager
                                                   -> whole-Expert Cache
                                                   -> ExpertTransferEngine
                                                        |
                          existing MultiDtypeWeightStore.prepare_unit()
                                                        |
                   existing pinned staging / copy stream / compute stream
                                                        |
                                          Naive or Grouped ExpertBackend
                                                        |
                                                   weighted Combine
```

## Weight objects and checkpoint adaptation

`WeightObjectKey(layer_id, kind, name, expert_id)` identifies Dense weights,
Embedding, LM Head, Router, Shared Expert and routed Expert objects without
changing physical `WeightSpec` identity. A `WeightObjectRecord` carries tensor
names, shapes, storage dtypes, byte counts and source-region offsets. Catalog
order is deterministic. Tied Embedding and LM Head retain separate semantic
objects while sharing the physical source.

`OlmoeModelAdapter` is the first family adapter. OLMoE-specific checkpoint
names terminate at that boundary. It creates:

- resident Router and norm placements;
- existing attention transfer units;
- one atomic transfer unit per complete gate/up/down Expert;
- an unchanged `ExecutionPlan` v1 plus `MoEExecutionSidecar` mappings.

For `OLMoE-1B-7B-0924`, metadata planning discovers 16 layers, 64 Experts per
layer, Top-8 routing, 1,024 complete Expert objects, 12 MiB per Expert and
12.00 GiB of Expert weights. The standard pinned-staging footprint is 24 MiB
for one slot or 48 MiB for two slots because the QKV transfer unit is larger
than one Expert. Vocabulary chunks do not inflate
Transformer/Expert slots.

Shared Experts, grouped routing and SSD sources are represented by the config
and catalog interfaces but are not executable qualification features yet.

## Router, dispatch and compute

`Router` applies the resident router GEMM, FP32 softmax, Top-K selection,
optional Top-K normalization and routed scaling. It returns device tensors in
`RoutingResult`.

`ExpertDispatcher` supports arbitrary token count `T`. The small-token path
avoids the general stable sort, while the general path creates sorted Expert
buckets and an inverse permutation. Dispatch and Combine remain tensorized.
The host `materialize_buckets()` method is explicitly a debug/reference bridge
and is not used by the grouped or streaming hot paths.

`NaiveExpertBackend` is the permanent correctness implementation. It accepts
the existing runtime's executable `LinearBackend` views. The dense
`GroupedExpertBackend` batches assignment-specific matrices with `torch.bmm`;
`AdaptiveExpertBackend` retains a selectable threshold instead of forcing
grouped execution for every shape.

## Residency, cache and ownership

`ExpertResidencyManager` owns metadata only. CPU/GPU allocation authority
remains in the existing WeightStore/runtime and the bounded
`ExpertDeviceArena`. Valid lifecycle states are:

```text
NOT_RESIDENT / CPU_RESIDENT
  -> H2D_INFLIGHT
  -> GPU_RESIDENT
  -> IN_USE
  -> EVICTABLE
  -> CPU_RESIDENT
```

Use count, pin count and inflight state independently protect an Expert from
eviction. Load submission is transactional and rolls back on exceptions.
Close/reset reject non-quiescent state.

`ExpertCache` stores one complete Expert per entry. Cache metadata records byte
size, location, ready event, admission/use timestamps, access count and loaded
bytes. LRU is the default; a frequency-aware policy is available. The cache
accepts only `UnifiedResidentWeightBudget(total, fixed_resident)`, so fixed
resident Router/norm/vocabulary weights and Expert cache capacity close under
the same `gpu-resident-weight-budget`. `MoEMemoryPlanner` composes the existing
`MemoryPlanner` estimate rather than altering its Dense API.

## Streaming and synchronization

`ExpertTransferEngine` does not own CPU checkpoint data or pinned staging. It
leases existing pipeline staging slots and calls the actual
`MultiDtypeWeightStore.prepare_unit()`. H2D is submitted to the existing copy
stream, records a ready event, and compute waits only on that event. There is
no `torch.cuda.synchronize()` or `cudaDeviceSynchronize()` in the MoE package.

The Scheduler performs one residency lookup over all selected Experts,
submits exact prefetch for misses, executes resident buckets first, then
consumes each missed Expert as its own ready event is reached. Profiling uses
CUDA events and computes copy/compute intersection with the existing timeline
accounting helper.

Dynamic GPU routing currently crosses one deliberately narrow, pinned host
bridge guarded by a CUDA Event because Python cache metadata needs concrete
Expert IDs. This is recorded as `routing_host_fence_ms`; it is not a device-wide
barrier. Removing this fence requires a device-resident scheduler/cache index
and is an M2 task.

## Continuous-batching-ready boundary

The implemented boundary is:

```text
Request Scheduler (future)
  -> Iteration hidden-state batch [T,H]
  -> Router [T,K]
  -> cross-token Expert buckets
  -> ExpertScheduler / cache
  -> Naive or Grouped backend
  -> Combine [T,H]
```

T=1, 2, 4, 8 and 32 are tested. Nothing inside Router, Dispatcher, grouped
compute or Combine assumes one request. A request scheduler and request/KV
ownership integration are intentionally out of scope.

## SSD future compatibility

The catalog source mapping and Expert Scheduler do not assume CPU RAM is the
only lower tier. An SSD-backed WeightStore may prepare a cold Expert into the
same pinned staging lease. This run had more than enough CPU RAM for the
qualification model, so no SSD critical-path code was added.
