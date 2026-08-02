#!/usr/bin/env python3
"""Long-run Fork/COW/Prefix/Beam/Speculative reference-count qualification."""

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming import KVPolicy, PagedKVRuntime  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--page-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--layers", type=int, default=2)
    return parser.parse_args()


def expected_references(runtime):
    counts = {}
    for state in runtime._requests.values():
        for handle in state.block_table.handles:
            counts[handle.identity()] = counts.get(handle.identity(), 0) + 1
    for handle in runtime._prefix_owned.values():
        counts[handle.identity()] = counts.get(handle.identity(), 0) + 1
    return counts


def verify_reference_counts(runtime):
    profile = runtime.quiesce()
    expected = expected_references(runtime)
    actual = {}
    for descriptor in runtime.page_pool.descriptors:
        if descriptor.ref_count:
            identity = (
                descriptor.page_id,
                descriptor.generation,
                descriptor.store_id,
                "{}:{}:{}".format(
                    descriptor.dtype,
                    descriptor.layout,
                    descriptor.format_version,
                ),
            )
            actual[identity] = descriptor.ref_count
        if descriptor.ref_count < 0 or descriptor.pin_count < 0:
            raise RuntimeError("negative page reference or pin count")
    if expected != actual:
        raise RuntimeError(
            "reference ownership mismatch: expected={}, actual={}".format(
                expected,
                actual,
            )
        )
    if profile["total_pin_count"]:
        raise RuntimeError("quiescent runtime retained page pins")
    if profile["kv_pool_allocated_pages"] != len(expected):
        raise RuntimeError("allocated page count does not match ownership graph")
    return profile


def append_tokens(runtime, state, token_count, layers, kv_heads, head_dim):
    key = torch.randn(
        token_count,
        kv_heads,
        head_dim,
        dtype=runtime.dtype,
        device=runtime.device,
    )
    value = torch.randn_like(key)
    for layer in range(layers):
        runtime.append((state,), layer, key, value, (token_count,))


def main():
    args = parse_args()
    if args.cycles <= 0 or args.sample_every <= 0 or args.layers <= 0:
        raise SystemExit("cycles, sample-every, and layers must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(913)
    head_dim = 128
    query_heads = 8
    kv_heads = 1
    max_length = args.cycles + args.page_size * 2
    page_count = int(math.ceil(max_length / float(args.page_size))) + 16
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    baseline_threads = len(threading.enumerate())
    snapshots = []
    token_history = [1]
    started = time.time()
    runtime = PagedKVRuntime(
        layer_count=args.layers,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        page_count=page_count,
        page_size=args.page_size,
        dtype=torch.bfloat16,
        device=device,
        policy=KVPolicy(
            dtype="bf16",
            page_size=args.page_size,
            reuse="prefix_memory",
            attention_backend="generic_cuda",
        ),
    )
    parent = runtime.create_request(
        max_length,
        request_id=1,
        reuse_namespace="lifecycle-soak",
    )
    append_tokens(runtime, parent, 1, args.layers, kv_heads, head_dim)
    try:
        for cycle in range(1, args.cycles + 1):
            # Beam branch is always rejected after exercising tail COW.
            beam = runtime.fork(parent, request_id=1000000 + cycle)
            append_tokens(runtime, beam, 1, args.layers, kv_heads, head_dim)
            runtime.discard_branch(beam)

            # Draft branch is accepted and atomically replaces parent state.
            draft = runtime.fork(parent, request_id=2000000 + cycle)
            append_tokens(runtime, draft, 1, args.layers, kv_heads, head_dim)
            runtime.commit_branch(parent, draft)
            token_history.append(4 + cycle)

            # Exercise session fork independently from speculative ownership.
            if cycle % 7 == 0:
                session = runtime.session_fork(
                    parent,
                    request_id=3000000 + cycle,
                )
                runtime.release(session)

            # Register and immediately look up every newly sealed prefix page.
            if cycle % args.page_size == args.page_size - 1:
                runtime.register_prefix(parent, token_history)
                reused, match = runtime.reuse_prefix(
                    token_history + [999999],
                    max_length=max_length,
                    reuse_namespace="lifecycle-soak",
                    request_id=4000000 + cycle,
                )
                if match.matched_tokens != (
                    len(token_history) // args.page_size * args.page_size
                ):
                    raise RuntimeError("prefix match length drifted")
                runtime.release(reused)

            if cycle == 1 or cycle % args.sample_every == 0:
                profile = verify_reference_counts(runtime)
                snapshots.append(
                    {
                        "cycle": cycle,
                        "sequence_length": parent.sequence_length,
                        "allocated_pages": profile["kv_pool_allocated_pages"],
                        "total_ref_count": profile["total_ref_count"],
                        "max_ref_count": profile["max_ref_count"],
                        "total_pin_count": profile["total_pin_count"],
                        "fork_count": profile["fork_count"],
                        "cow_count": profile["cow_count"],
                        "prefix_hits": profile["prefix_hits"],
                        "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                        "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
                        "thread_count": len(threading.enumerate()),
                    }
                )
        final_profile = verify_reference_counts(runtime)
        runtime.release(parent)
        post_request_profile = verify_reference_counts(runtime)
    finally:
        runtime.close()
    torch.cuda.synchronize(device)
    post_close_profile = runtime.page_pool.profile()
    final_allocated = torch.cuda.memory_allocated(device)
    final_reserved = torch.cuda.memory_reserved(device)
    final_threads = len(threading.enumerate())
    acceptance = {
        "ownership_graph_exact_every_sample": True,
        "quiescent_pin_count_zero": all(
            item["total_pin_count"] == 0 for item in snapshots
        ),
        "all_pages_released_after_close": (
            post_close_profile["allocated_pages"] == 0
        ),
        "thread_count_stable": final_threads == baseline_threads,
        "cuda_allocated_returned": (
            final_allocated - baseline_allocated <= 8 * 1024 * 1024
        ),
        "cuda_reserved_bounded": (
            final_reserved - baseline_reserved <= 64 * 1024 * 1024
        ),
    }
    report = {
        "schema_version": 1,
        "cycles": args.cycles,
        "page_size": args.page_size,
        "layers": args.layers,
        "duration_seconds": time.time() - started,
        "operations": {
            "beam_forks_and_discards": args.cycles,
            "speculative_forks_and_commits": args.cycles,
            "session_forks": args.cycles // 7,
            "prefix_registrations": args.cycles // args.page_size,
        },
        "final_active_profile": final_profile,
        "post_request_profile": post_request_profile,
        "post_close_pool_profile": post_close_profile,
        "resource_drift": {
            "cuda_allocated_bytes": final_allocated - baseline_allocated,
            "cuda_reserved_bytes": final_reserved - baseline_reserved,
            "thread_count": final_threads - baseline_threads,
        },
        "acceptance": acceptance,
        "all_acceptance_passed": all(acceptance.values()),
        "snapshots": snapshots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["all_acceptance_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
