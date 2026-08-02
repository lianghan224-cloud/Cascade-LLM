#!/usr/bin/env python3
"""Long-running end-to-end stability test for the KV Framework V1 runtime."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time

import torch
from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming import (  # noqa: E402
    ExecutionPolicy,
    KVPolicy,
    Llama31DecodeExecutor,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MixedVocabStreamingRuntime,
    MultiDtypeWeightStore,
    PagedKVRuntime,
    PlacementMode,
    RequestKVCacheV1,
    adapter_for_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-tokens", type=int, default=1000)
    parser.add_argument("--load-cycles", type=int, default=1)
    parser.add_argument("--allocation-cycles", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--page-size", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--provider",
        choices=("generic_cuda", "sm80", "sm86", "sm89", "sm90"),
        default="generic_cuda",
    )
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="pinned_staging",
    )
    parser.add_argument("--cuda-allocated-drift-mib", type=float, default=8.0)
    parser.add_argument("--cuda-reserved-drift-mib", type=float, default=64.0)
    return parser.parse_args()


def snapshot(index, token, latency, kv_runtime):
    profile = kv_runtime.profile_stats()
    return {
        "sample": index,
        "token": token,
        "latency_ms": latency,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(kv_runtime.device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(kv_runtime.device),
        "thread_count": len(threading.enumerate()),
        "free_pages": profile["page_state_counts"]["free"],
        "allocated_pages": profile["kv_pool_allocated_pages"],
        "event_count": profile["cuda_event_count"],
        "attention_calls": profile["attention_calls"],
        "workspace_peak_bytes": profile["workspace_peak_bytes"],
    }


def main():
    args = parse_args()
    for name in ("decode_tokens", "load_cycles", "allocation_cycles", "sample_every"):
        if getattr(args, name) <= 0:
            raise SystemExit("--{} must be positive".format(name.replace("_", "-")))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if args.decode_tokens > config.max_position_embeddings:
        raise SystemExit("decode token count exceeds model max_position_embeddings")
    inferred = ExecutionPolicy.from_config(config)
    policy = ExecutionPolicy.from_config(
        config,
        cpu_weight_mode=args.weight_store,
        granularity="matrix_group",
        embedding_mode=PlacementMode.STREAMED,
        lm_head_mode=PlacementMode.STREAMED,
        slot_count=2,
        prefetch_depth=2,
        vocab_chunk_bytes=4 * 1024 * 1024,
        weight_format=inferred.weight_format,
        quantization=inferred.quantization,
    )
    adapter = adapter_for_config(config)
    plan = adapter.build_execution_plan(config, policy)
    adapter.validate_checkpoint(args.checkpoint, config, policy).raise_for_error()
    compute_dtype = next(
        spec.compute_dtype
        for spec in plan.weights.values()
        if spec.role == "attention_q"
    )
    dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16
    page_count = int(math.ceil(args.decode_tokens / float(args.page_size)))
    all_snapshots = []
    all_latencies = []
    baseline_threads = len(threading.enumerate())
    started = time.time()
    for load_cycle in range(args.load_cycles):
        with ExitStack() as resources:
            store = resources.enter_context(
                MultiDtypeWeightStore(plan, args.weight_store, staging_slot_count=2)
            )
            store.load_checkpoint(args.checkpoint)
            resident = resources.enter_context(
                MixedResidentDeviceArena(plan, store, device)
            )
            runtime = resources.enter_context(
                MixedDtypeRuntime(plan, store, resident, device, slot_count=2)
            )
            vocab = resources.enter_context(
                MixedVocabStreamingRuntime(
                    plan, store, runtime, embedding_staging_rows=1
                )
            )
            kv_runtime = resources.enter_context(
                PagedKVRuntime(
                    layer_count=config.num_hidden_layers,
                    num_query_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                    page_count=page_count,
                    page_size=args.page_size,
                    dtype=dtype,
                    device=device,
                    policy=KVPolicy(
                        dtype=("bf16" if dtype == torch.bfloat16 else "fp16"),
                        page_size=args.page_size,
                        attention_backend=args.provider,
                    ),
                )
            )
            request = kv_runtime.create_request(args.decode_tokens)
            cache = RequestKVCacheV1(kv_runtime, request)
            executor = Llama31DecodeExecutor(
                config,
                resident,
                kv_cache=cache,
                vocab_runtime=vocab,
                top_k=min(10, config.vocab_size),
            )
            try:
                with torch.inference_mode():
                    for token in range(args.decode_tokens):
                        token_id = 1 if token == 0 else 4 + token % max(1, config.vocab_size - 4)
                        input_ids = torch.tensor([[token_id]], device=device)
                        token_started = time.perf_counter()
                        state = executor.finish(
                            runtime.run(executor, executor.begin(input_ids))
                        )
                        del state
                        torch.cuda.synchronize(device)
                        latency = (time.perf_counter() - token_started) * 1000.0
                        all_latencies.append(latency)
                        if (token + 1) % args.sample_every == 0 or token == 0:
                            all_snapshots.append(
                                snapshot(
                                    len(all_snapshots), token + 1, latency, kv_runtime
                                )
                            )
            finally:
                executor.close()
                cache.close()
            if kv_runtime.page_pool.allocated_pages:
                raise RuntimeError("request release left KV pages allocated")
            for cycle in range(args.allocation_cycles):
                state = kv_runtime.create_request(
                    args.page_size,
                    request_id=1000000 + load_cycle * args.allocation_cycles + cycle,
                )
                key = torch.zeros(
                    1, config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                    dtype=dtype, device=device,
                )
                for layer in range(config.num_hidden_layers):
                    kv_runtime.append((state,), layer, key, key, (1,))
                kv_runtime.release(state)
            if kv_runtime.page_pool.allocated_pages:
                raise RuntimeError("allocation cycles left KV pages allocated")
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    final_threads = len(threading.enumerate())
    if final_threads != baseline_threads:
        raise RuntimeError(
            "thread count changed from {} to {}".format(
                baseline_threads, final_threads
            )
        )
    first = all_snapshots[0]
    last = all_snapshots[-1]
    allocated_drift = last["cuda_allocated_bytes"] - first["cuda_allocated_bytes"]
    reserved_drift = last["cuda_reserved_bytes"] - first["cuda_reserved_bytes"]
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "provider": args.provider,
        "decode_tokens_per_cycle": args.decode_tokens,
        "load_cycles": args.load_cycles,
        "allocation_cycles_per_load": args.allocation_cycles,
        "page_size": args.page_size,
        "duration_seconds": time.time() - started,
        "latency_ms": {
            "mean": statistics.mean(all_latencies),
            "p50": statistics.median(all_latencies),
            "max": max(all_latencies),
            "first": all_latencies[0],
            "last": all_latencies[-1],
        },
        "resource_drift": {
            "cuda_allocated_bytes": allocated_drift,
            "cuda_reserved_bytes": reserved_drift,
            "thread_count": final_threads - baseline_threads,
        },
        "acceptance": {
            "cuda_allocated_stable": allocated_drift <= args.cuda_allocated_drift_mib * 1024 * 1024,
            "cuda_reserved_stable": reserved_drift <= args.cuda_reserved_drift_mib * 1024 * 1024,
            "thread_count_stable": final_threads == baseline_threads,
            "workspace_zero": all(item["workspace_peak_bytes"] == 0 for item in all_snapshots),
            "all_pages_released": True,
        },
        "snapshots": all_snapshots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
