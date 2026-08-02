#!/usr/bin/env python3
"""Experimental D0/D1 legacy KV ablation; not a production benchmark."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming.experimental import (  # noqa: E402
    DensePagedAttention,
    DensePagedOnlineAttention,
    KVCacheManager,
)
from layer_streaming import KVPolicy  # noqa: E402


REPORT_SCHEMA_VERSION = 1


def _csv_ints(value):
    try:
        result = tuple(int(item.strip()) for item in str(value).split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--contexts", type=_csv_ints, default=(128, 512, 2048))
    parser.add_argument("--page-sizes", type=_csv_ints, default=(16, 32))
    parser.add_argument("--phase", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--paged-backend",
        choices=("sdpa", "online", "both"),
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _measure(callback, device, warmup, repeats):
    for _ in range(int(warmup)):
        callback()
    _synchronize(device)
    samples = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        callback()
        _synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.pstdev(samples),
    }


def run_case(
    *,
    device,
    context,
    page_size,
    phase,
    attention_heads,
    kv_heads,
    head_dim,
    dtype,
    warmup,
    repeats,
    seed,
    paged_backend,
):
    device = torch.device(device)
    if attention_heads % kv_heads:
        raise ValueError("attention_heads must be divisible by kv_heads")
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    query_tokens = int(context) if phase == "prefill" else 1
    query = torch.randn(
        (1, attention_heads, query_tokens, head_dim),
        dtype=torch_dtype,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        (1, kv_heads, context, head_dim),
        dtype=torch_dtype,
        device=device,
        generator=generator,
    )
    value = torch.randn(
        (1, kv_heads, context, head_dim),
        dtype=torch_dtype,
        device=device,
        generator=generator,
    )
    groups = attention_heads // kv_heads
    repeated_key = key.repeat_interleave(groups, dim=1)
    repeated_value = value.repeat_interleave(groups, dim=1)
    positions = (
        torch.arange(context, dtype=torch.long, device=device).unsqueeze(0)
        if phase == "prefill"
        else torch.tensor([[context - 1]], dtype=torch.long, device=device)
    )
    total_pages = int(math.ceil(context / float(page_size)))
    attention_backend = (
        DensePagedAttention()
        if paged_backend == "sdpa"
        else DensePagedOnlineAttention()
    )
    manager = KVCacheManager(
        layer_count=1,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        total_blocks=total_pages,
        block_size=page_size,
        dtype=torch_dtype,
        device=device,
        policy=KVPolicy(dtype=dtype, page_size=page_size),
        attention_backend=attention_backend,
    )
    cache = manager.bind(manager.allocate(context))
    append_started = time.perf_counter()
    cache.append_only(0, key, value)
    _synchronize(device)
    append_ms = (time.perf_counter() - append_started) * 1000.0

    def continuous():
        return F.scaled_dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            dropout_p=0.0,
            is_causal=(phase == "prefill"),
        )

    def paged():
        return cache.attend(
            0,
            query,
            kv_groups=groups,
            position_ids=positions,
        )

    with torch.inference_mode():
        reference = continuous()
        candidate = paged()
        fp32_reference = F.scaled_dot_product_attention(
            query.float(),
            repeated_key.float(),
            repeated_value.float(),
            dropout_p=0.0,
            is_causal=(phase == "prefill"),
        )
        _synchronize(device)
        difference = (reference.float() - candidate.float()).abs()
        continuous_fp32_error = (reference.float() - fp32_reference).abs()
        paged_fp32_error = (candidate.float() - fp32_reference).abs()
        manager.reset_profile()
        continuous_timing = _measure(continuous, device, warmup, repeats)
        paged_timing = _measure(paged, device, warmup, repeats)
    profile = manager.profile_stats()
    result = {
        "case": {
            "phase": phase,
            "context": int(context),
            "page_size": int(page_size),
            "attention_heads": int(attention_heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
            "dtype": dtype,
            "device": str(device),
            "paged_backend": attention_backend.name,
        },
        "correctness": {
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "allclose_atol_4e_3_rtol_4e_3": bool(
                torch.allclose(reference, candidate, atol=4e-3, rtol=4e-3)
            ),
            "continuous_vs_fp32_max_abs_error": float(
                continuous_fp32_error.max().item()
            ),
            "continuous_vs_fp32_mean_abs_error": float(
                continuous_fp32_error.mean().item()
            ),
            "paged_vs_fp32_max_abs_error": float(
                paged_fp32_error.max().item()
            ),
            "paged_vs_fp32_mean_abs_error": float(
                paged_fp32_error.mean().item()
            ),
        },
        "performance": {
            "append_ms": append_ms,
            "continuous_sdpa": continuous_timing,
            "paged_reference": paged_timing,
            "paged_over_continuous_ratio": (
                paged_timing["median_ms"] / continuous_timing["median_ms"]
                if continuous_timing["median_ms"] > 0
                else None
            ),
        },
        "memory": {
            "continuous_kv_bytes": key.numel() * key.element_size() * 2,
            "paged_pool_bytes": manager.nbytes,
            "rounded_page_capacity_tokens": total_pages * page_size,
            "materialized_full_kv_bytes": profile["materialized_bytes"],
        },
        "paged_profile": profile,
    }
    cache.close()
    manager.close()
    return result


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup must be non-negative and repeats must be positive")
    phases = ("prefill", "decode") if args.phase == "both" else (args.phase,)
    paged_backends = (
        ("sdpa", "online")
        if args.paged_backend == "both"
        else (args.paged_backend,)
    )
    results = []
    for context in args.contexts:
        for page_size in args.page_sizes:
            for phase in phases:
                for paged_backend in paged_backends:
                    results.append(
                        run_case(
                            device=args.device,
                            context=context,
                            page_size=page_size,
                            phase=phase,
                            attention_heads=args.attention_heads,
                            kv_heads=args.kv_heads,
                            head_dim=args.head_dim,
                            dtype=args.dtype,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            seed=args.seed,
                            paged_backend=paged_backend,
                        )
                    )
    device = torch.device(args.device)
    hardware = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware.update(
            {
                "gpu_name": properties.name,
                "compute_capability": list(
                    torch.cuda.get_device_capability(device)
                ),
                "total_memory_bytes": properties.total_memory,
            }
        )
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": "synthetic D0/D1 attention ablation; not a real-model performance claim",
        "hardware": hardware,
        "implemented": [
            "orthogonal KV policy contract",
            "HND exact GPU page pool",
            "request block table",
            "fork and copy-on-write",
            "page-wise dense exact reference attention",
            "materialized SDPA correctness reference attention",
        ],
        "known_defects": [
            "both paged references use Python page loops and are not fused production kernels",
            "dense_paged_sdpa_reference copies a full layer KV view and requires explicit workspace",
            "CUDA timing is wall-clock synchronized rather than provider event timing",
            "cross-request prefix lookup, CPU/NVMe tiers, quantized KV, and sparse indexes are not implemented",
            "real-model 4K-128K quality and performance qualification remains pending",
        ],
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
