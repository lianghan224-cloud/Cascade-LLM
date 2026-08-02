#!/usr/bin/env python3
"""Benchmark KV Attention modes inside the real streamed 8B execution path."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming import (  # noqa: E402
    ExecutionPolicy,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MultiDtypeWeightStore,
    PlacementMode,
    adapter_for_config,
)
from tools.qualify_kv_quality import QualityRunner  # noqa: E402


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--providers",
        type=csv_strings,
        default=("legacy_gather_sdpa_reference", "generic_cuda", "sm86"),
    )
    parser.add_argument("--lengths", type=csv_ints, default=(8, 128, 512, 2048))
    parser.add_argument("--page-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--slots", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="pinned_staging",
    )
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(float(item) for item in values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(values):
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples_ms": values,
    }


def make_prompt(tokenizer, length):
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = 1
    vocab_size = len(tokenizer)
    return [bos] + [4 + (index % max(1, vocab_size - 4)) for index in range(length - 1)]


def run_once(runner, provider, prompt, decode_token):
    torch.cuda.reset_peak_memory_stats(runner.device)
    with runner.executor(provider, len(prompt) + 1) as executor:
        started = time.perf_counter()
        prefill_logits = runner.forward(executor, prompt)
        torch.cuda.synchronize(runner.device)
        ttft_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        decode_logits = runner.forward(executor, (decode_token,))
        torch.cuda.synchronize(runner.device)
        decode_ms = (time.perf_counter() - started) * 1000.0
        profile = executor.kv_cache.profile_stats()
        result = {
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "prefill_top1": int(prefill_logits.argmax(dim=-1).item()),
            "decode_top1": int(decode_logits.argmax(dim=-1).item()),
            "workspace_peak_bytes": int(profile["workspace_peak_bytes"]),
            "kv_pool_bytes": int(profile["store_bytes"]),
            "gpu_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(runner.device)
            ),
            "provider_fallback_reason": profile["provider_fallback_reason"],
            "attention_backend": profile["attention_backend"],
            "page_kernel_backend": profile["paged_kv_kernel_backend"],
        }
    return result


def main():
    args = parse_args()
    if args.warmup < 0 or args.runs <= 0:
        raise SystemExit("warmup must be non-negative and runs must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    inferred = ExecutionPolicy.from_config(config)
    policy = ExecutionPolicy(
        granularity="matrix_group",
        weight_format=inferred.weight_format,
        cpu_weight_mode=args.weight_store,
        embedding_mode=PlacementMode.RESIDENT,
        lm_head_mode=PlacementMode.RESIDENT,
        slot_count=args.slots,
        prefetch_depth=args.slots,
        embedding_dtype=inferred.embedding_dtype,
        lm_head_dtype=inferred.lm_head_dtype,
        norm_dtype=inferred.norm_dtype,
        quantization=inferred.quantization,
        linear_backend=None,
    )
    adapter = adapter_for_config(config)
    plan = adapter.build_execution_plan(config, policy)
    adapter.validate_checkpoint(args.checkpoint, config, policy).raise_for_error()
    results = []
    with ExitStack() as resources:
        store = resources.enter_context(
            MultiDtypeWeightStore(
                plan, args.weight_store, staging_slot_count=args.slots
            )
        )
        store.load_checkpoint(args.checkpoint)
        resident = resources.enter_context(
            MixedResidentDeviceArena(plan, store, device)
        )
        runtime = resources.enter_context(
            MixedDtypeRuntime(
                plan, store, resident, device, slot_count=args.slots
            )
        )
        runner = QualityRunner(
            config, plan, resident, runtime, tokenizer, device, args.page_size
        )
        for length in args.lengths:
            prompt = make_prompt(tokenizer, int(length))
            decode_token = 4 + int(length) % max(1, len(tokenizer) - 4)
            for provider in args.providers:
                for _ in range(args.warmup):
                    run_once(runner, provider, prompt, decode_token)
                samples = [
                    run_once(runner, provider, prompt, decode_token)
                    for _ in range(args.runs)
                ]
                ttft = summarize([item["ttft_ms"] for item in samples])
                decode = summarize([item["decode_ms"] for item in samples])
                item = {
                    "provider": provider,
                    "context_length": int(length),
                    "page_size": args.page_size,
                    "runs": args.runs,
                    "ttft": ttft,
                    "decode": decode,
                    "decode_tokens_per_second": 1000.0 / decode["p50_ms"],
                    "workspace_peak_bytes": max(
                        sample["workspace_peak_bytes"] for sample in samples
                    ),
                    "kv_pool_bytes": max(
                        sample["kv_pool_bytes"] for sample in samples
                    ),
                    "gpu_peak_allocated_bytes": max(
                        sample["gpu_peak_allocated_bytes"] for sample in samples
                    ),
                    "prefill_top1": [sample["prefill_top1"] for sample in samples],
                    "decode_top1": [sample["decode_top1"] for sample in samples],
                    "provider_fallback_reason": samples[-1]["provider_fallback_reason"],
                    "attention_backend": samples[-1]["attention_backend"],
                    "page_kernel_backend": samples[-1]["page_kernel_backend"],
                }
                results.append(item)
                print(
                    "real8b provider={} context={} ttft_p50={:.3f} "
                    "decode_p50={:.3f}".format(
                        provider,
                        length,
                        ttft["p50_ms"],
                        decode["p50_ms"],
                    ),
                    flush=True,
                )
    by_length = {}
    for item in results:
        by_length.setdefault(item["context_length"], {})[item["provider"]] = item
    for length, providers in by_length.items():
        reference = providers.get("legacy_gather_sdpa_reference")
        if reference is None:
            continue
        for item in providers.values():
            item["decode_speedup_vs_legacy"] = (
                reference["decode"]["p50_ms"] / item["decode"]["p50_ms"]
            )
            item["ttft_speedup_vs_legacy"] = (
                reference["ttft"]["p50_ms"] / item["ttft"]["p50_ms"]
            )
            item["top1_matches_legacy"] = (
                item["prefill_top1"] == reference["prefill_top1"]
                and item["decode_top1"] == reference["decode_top1"]
            )
    report = {
        "schema_version": 1,
        "benchmark": "real8b_kv_mode_matrix",
        "checkpoint": str(args.checkpoint.resolve()),
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "architecture": "sm{}{}".format(
                *torch.cuda.get_device_capability(device)
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "configuration": {
            "providers": list(args.providers),
            "lengths": list(args.lengths),
            "page_size": args.page_size,
            "warmup": args.warmup,
            "runs": args.runs,
            "weight_store": args.weight_store,
            "slots": args.slots,
            "embedding_placement": "resident",
            "lm_head_placement": "resident",
            "transformer_placement": "streamed",
        },
        "results": results,
        "interpretation": (
            "TTFT and decode include streamed Transformer execution and the "
            "resident LM Head; kernel-only speedups must not be substituted "
            "for these end-to-end values."
        ),
    }
    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
