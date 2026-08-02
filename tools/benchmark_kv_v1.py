#!/usr/bin/env python3
"""Reproducible KV Framework V1 provider and page-layout ablation."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming import (  # noqa: E402
    KVPolicy,
    PagedKVRuntime,
    default_paged_numerical_contract,
)


def csv_ints(value):
    result = tuple(int(item) for item in str(value).split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def csv_strings(value):
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--providers", type=csv_strings, default=(
        "reference_paged_exact", "generic_cuda", "sm86"
    ))
    parser.add_argument("--lengths", type=csv_ints, default=(16, 128, 512))
    parser.add_argument("--page-sizes", type=csv_ints, default=(16, 32))
    parser.add_argument("--phases", type=csv_strings, default=("decode", "prefill"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prefill-query-length", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def run_case(args, provider, context_length, page_size, phase, tensors):
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    lengths, key, value, query, query_lengths, positions = tensors
    pages = sum(int(math.ceil(item / float(page_size))) for item in lengths) + 4
    torch.cuda.reset_peak_memory_stats(device)
    runtime = None
    try:
        runtime = PagedKVRuntime(
            layer_count=1,
            num_query_heads=args.query_heads,
            num_kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            page_count=pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            policy=KVPolicy(
                dtype=args.dtype,
                page_size=page_size,
                attention_backend=provider,
            ),
            allow_reference=provider in {
                "reference_paged_exact", "legacy_gather_sdpa_reference"
            },
        )
        requests = tuple(
            runtime.create_request(length, request_id=index + 1)
            for index, length in enumerate(lengths)
        )
        runtime.append(requests, 0, key, value, lengths)
        for _ in range(args.warmup):
            runtime.attend(
                requests, 0, query, query_lengths,
                query_positions=positions, phase=phase,
            )
        torch.cuda.synchronize(device)
        samples = []
        output = None
        for _ in range(args.runs):
            started = time.perf_counter()
            output = runtime.attend(
                requests, 0, query, query_lengths,
                query_positions=positions, phase=phase,
            ).output
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
        profile = runtime.profile_stats()
        return {
            "supported": True,
            "provider": provider,
            "context_length": context_length,
            "page_size": page_size,
            "phase": phase,
            "batch_size": len(lengths),
            "query_lengths": list(query_lengths),
            "mean_ms": statistics.mean(samples),
            "p50_ms": statistics.median(samples),
            "p95_ms": _percentile(samples, 0.95),
            "samples_ms": samples,
            "workspace_peak_bytes": profile["workspace_peak_bytes"],
            "kv_pool_bytes": profile["store_bytes"],
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "requires_full_kv_workspace": runtime.provider.capability().requires_full_kv_workspace,
            "qualification_status": runtime.provider.capability().qualification_status,
            "attention_backend": runtime.attention_backend.name,
            "kv_kernel_backend": runtime.kv_kernel_backend.name,
            "provider_bundle": runtime.provider_bundle.name,
            "output": output.detach(),
        }
    except Exception as error:
        return {
            "supported": False,
            "provider": provider,
            "context_length": context_length,
            "page_size": page_size,
            "phase": phase,
            "unsupported_reason": "{}: {}".format(type(error).__name__, error),
        }
    finally:
        if runtime is not None:
            runtime.close()
        torch.cuda.synchronize(device)


def build_tensors(args, context_length, phase):
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    lengths = tuple(
        max(1, context_length - index * max(1, context_length // (2 * args.batch_size)))
        for index in range(args.batch_size)
    )
    query_lengths = tuple(
        1 if phase == "decode" else min(args.prefill_query_length, length)
        for length in lengths
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(20260802 + context_length + (0 if phase == "decode" else 10000))
    key = torch.randn(
        sum(lengths), args.kv_heads, args.head_dim,
        dtype=dtype, device=device, generator=generator,
    )
    value = torch.randn_like(key)
    query = torch.randn(
        sum(query_lengths), args.query_heads, args.head_dim,
        dtype=dtype, device=device, generator=generator,
    )
    positions = tuple(
        torch.arange(length - query_length, length, device=device)
        for length, query_length in zip(lengths, query_lengths)
    )
    return lengths, key, value, query, query_lengths, positions


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    architecture = "sm{}{}".format(*torch.cuda.get_device_capability(device))
    results = []
    for context_length in args.lengths:
        for page_size in args.page_sizes:
            for phase in args.phases:
                tensors = build_tensors(args, context_length, phase)
                case = [
                    run_case(
                        args, provider, context_length, page_size, phase, tensors
                    )
                    for provider in args.providers
                ]
                reference = next(
                    (
                        item for item in case
                        if item["supported"]
                        and item["provider"] == "reference_paged_exact"
                    ),
                    None,
                )
                generic = next(
                    (
                        item for item in case
                        if item["supported"] and item["provider"] == "generic_cuda"
                    ),
                    None,
                )
                reference_output = (
                    None if reference is None else reference.get("output")
                )
                reference_p50 = (
                    None if reference is None else reference["p50_ms"]
                )
                generic_p50 = None if generic is None else generic["p50_ms"]
                for item in case:
                    output = item.pop("output", None)
                    if output is not None and reference_output is not None:
                        contract = default_paged_numerical_contract(
                            architecture, item["provider"], args.dtype
                        )
                        item["numerical_contract"] = contract.evaluate(
                            reference_output, output
                        )
                        item["numerical_contract"]["contract"] = contract.as_dict()
                    if reference_p50 is not None and item["supported"]:
                        item["speedup_vs_reference"] = (
                            reference_p50 / item["p50_ms"]
                        )
                    if generic_p50 is not None and item["supported"]:
                        item["speedup_vs_generic"] = generic_p50 / item["p50_ms"]
                results.extend(case)
    production = [
        item for item in results
        if item["supported"] and item["provider"] not in {
            "reference_paged_exact", "legacy_gather_sdpa_reference"
        }
    ]
    has_reference = any(
        item["supported"] and item["provider"] == "reference_paged_exact"
        for item in results
    )
    report = {
        "schema_version": 1,
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "architecture": architecture,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "configuration": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "results": results,
        "acceptance": {
            "production_numerics_pass": (None if not has_reference else bool(production) and all(
                item.get("numerical_contract", {}).get("passed", False)
                for item in production
            )),
            "production_has_zero_full_kv_workspace": bool(production) and all(
                item["workspace_peak_bytes"] == 0
                and not item["requires_full_kv_workspace"]
                for item in production
            ),
            "production_faster_than_reference": (None if not has_reference else bool(production) and all(
                item.get("speedup_vs_reference", 0.0) > 1.0
                for item in production
            )),
        },
    }
    text = json.dumps(report, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
