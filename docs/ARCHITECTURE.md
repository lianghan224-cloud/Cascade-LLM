# Cascade-LLM Architecture

Cascade-LLM targets a single machine with large CPU RAM and a GPU whose VRAM is
smaller than the model checkpoint. The runtime is deliberately explicit about
where data resides and which component owns each resource.

```text
Checkpoint metadata -> Model adapter -> Memory planner
                                      -> CPU weight arena
Request/session -> model executor -> Weight runtime -> GPU slots / streams
                                  -> KV runtime     -> paged KV arena
                                  -> backend        -> provider kernels
```

## Runtime ownership

- **Model adapter** derives geometry, tensor mappings, and execution metadata
  from checkpoint configuration. Model shapes are never hard-coded by size.
- **Memory planner** performs the sole GPU and host-memory feasibility check
  before CPU arena or GPU allocation.
- **Weight runtime** owns typed CPU regions, pinned staging, bounded device
  slots, copy/compute streams, and transfer fences.
- **KV runtime** owns logical block tables, page generations, reference and
  pin counts, prefix references, copy-on-write, and rollback.
- **Backend/provider** performs computation only. It never mutates KV or
  weight ownership metadata.

## Execution model

The public dense path validates metadata and checkpoint tensors, runs the
memory preflight, creates CPU arenas, initializes bounded GPU resources, then
executes prefill and decode. Weight transfer may be matrix-, matrix-group-, or
layer-granular. The selected policy and every fallback are recorded in the run
report.

Resident embedding, LM head, norms, and complete Transformer-layer placement
are optional and included in the same memory preflight as transfer slots,
workspace, activations, and KV capacity.

## Extension boundaries

MoE uses an additive weight-object catalog and expert residency layer over the
same CPU store, staging slots, streams, planner, and linear backends used by
dense execution. It must not introduce a second weight-store ownership model.

Paged KV exposes a batch-oriented view, but service-level continuous batching
is not qualified as a public runtime feature. See [STATUS.md](STATUS.md) for
the evidence boundary.
