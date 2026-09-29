# Current Status

## Overall

Cascade-LLM is a **research prototype**. It has demonstrated short dense
single-GPU runs with model weights substantially larger than VRAM, but it is
not production qualified and is not yet a general-purpose serving system.

## Implemented and testable

- Llama-family checkpoint metadata validation, CPU weight arenas, bounded
  staged transfer, GPU memory preflight, and dense decode execution.
- Explicit BF16/FP16 and INT8/INT4 fallback paths, with capability-gated
  SM86 W8A16 provider support.
- Paged KV ownership with page generations, reference/pin accounting,
  copy-on-write, prefix reuse, rollback, and exact reference routing.
- MoE building blocks: weight-object catalog, router, dispatch, expert cache,
  and streaming primitives. Dense execution remains on its dense hot path.

## Not qualified or incomplete

- Real 70B W8A16/W4A16 correctness and end-to-end performance qualification.
- Long-running, exclusive-GPU dense qualification at long context lengths.
- Continuous batching, request admission, cancellation lifecycle, and a
  long-lived multi-request serving daemon.
- Real-checkpoint MoE qualification and measured expert-cache behavior.
- Real NVMe/GDS KV data plane, production Active Tier behavior, and RGKV
  quality qualification.

## Evidence rules

`FUNCTIONAL`, `CUDA_SMOKE`, and synthetic tests are not real-model performance
qualification. A blocked hardware, checkpoint, or exclusive-GPU experiment is
reported as blocked, never as a pass. Historical reports can be useful context
but do not supersede the contracts and implementation currently in the tree.
