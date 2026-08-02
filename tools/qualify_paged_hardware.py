#!/usr/bin/env python3
"""Execute paged-provider qualification on the physically available GPU."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming import (  # noqa: E402
    KVPolicy,
    PagedKVRuntime,
    default_paged_numerical_contract,
    default_paged_registry,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def run_case(device, provider, query_heads, kv_heads, phase, runs):
    dtype = torch.bfloat16
    length = 33
    query_length = length if phase == "prefill" else 1
    runtime = PagedKVRuntime(
        layer_count=1,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=128,
        page_count=8,
        page_size=16,
        dtype=dtype,
        device=device,
        policy=KVPolicy(
            dtype="bf16",
            page_size=16,
            attention_backend=provider,
        ),
    )
    try:
        state = runtime.create_request(64)
        generator = torch.Generator(device=device)
        generator.manual_seed(9000 + query_heads * 10 + kv_heads)
        key = torch.randn(
            length,
            kv_heads,
            128,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        value = torch.randn_like(key)
        query = torch.randn(
            query_length,
            query_heads,
            128,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        runtime.append((state,), 0, key, value, (length,))
        positions = (
            torch.arange(length, device=device)
            if phase == "prefill"
            else torch.tensor([length - 1], device=device)
        )
        for _ in range(2):
            runtime.attend(
                (state,),
                0,
                query,
                (query_length,),
                query_positions=(positions,),
                phase=phase,
            )
        torch.cuda.synchronize(device)
        samples = []
        output = None
        for _ in range(runs):
            started = time.perf_counter()
            output = runtime.attend(
                (state,),
                0,
                query,
                (query_length,),
                query_positions=(positions,),
                phase=phase,
            ).output
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
        groups = query_heads // kv_heads
        expected = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, 1),
            value.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, 1),
            dropout_p=0.0,
            is_causal=(phase == "prefill"),
        ).squeeze(0).transpose(0, 1)
        architecture = "sm{}{}".format(
            *torch.cuda.get_device_capability(device)
        )
        contract = default_paged_numerical_contract(
            architecture,
            provider,
            "bf16",
        )
        numerical = contract.evaluate(expected, output)
        profile = runtime.quiesce()
        return {
            "phase": phase,
            "attention_kind": (
                "mqa" if kv_heads == 1
                else "mha" if kv_heads == query_heads
                else "gqa"
            ),
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "context_length": length,
            "mean_ms": sum(samples) / len(samples),
            "samples_ms": samples,
            "numerical_contract": numerical,
            "workspace_bytes": profile["workspace_peak_bytes"],
            "attention_backend": profile["paged_attention_provider"],
            "kv_kernel_backend": profile["paged_kv_kernel_backend"],
            "provider_bundle": profile["paged_provider_bundle"],
            "fallback_reason": profile["provider_fallback_reason"],
        }
    finally:
        runtime.close()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    detected = "sm{}{}".format(*torch.cuda.get_device_capability(device))
    registry = default_paged_registry()
    capabilities = registry.capabilities()
    providers = {}
    for target in ("sm80", "sm86", "sm89", "sm90"):
        capability = capabilities.get(target)
        item = {
            "provider": target,
            "bundle_load_path_present": capability is not None,
            "attention_backend": (
                None if capability is None else capability["attention_backend"]
            ),
            "kv_kernel_backend": (
                None if capability is None else capability["kv_kernel_backend"]
            ),
            "declared_capability": capability,
            "physical_hardware_available": target == detected,
            "executed": False,
            "cases": [],
        }
        if capability is None:
            item.update(
                status="unsupported",
                unqualified_reason="provider bundle is not loadable",
            )
        elif target != detected:
            item.update(
                status="unqualified",
                unqualified_reason=(
                    "no physical {} GPU is installed; cross-architecture "
                    "execution and numerical qualification were not run"
                ).format(target),
            )
        else:
            cases = []
            for query_heads, kv_heads in ((8, 8), (8, 2), (8, 1)):
                for phase in ("prefill", "decode"):
                    cases.append(
                        run_case(
                            device,
                            target,
                            query_heads,
                            kv_heads,
                            phase,
                            args.runs,
                        )
                    )
            passed = all(
                case["numerical_contract"]["passed"]
                and case["workspace_bytes"] == 0
                and case["fallback_reason"] is None
                for case in cases
            )
            item.update(
                executed=True,
                cases=cases,
                status=("smoke_passed" if passed else "failed"),
                unqualified_reason=(
                    "real-model and 1000-token evidence must be aggregated "
                    "before qualified status"
                    if passed
                    else "one or more executable provider cases failed"
                ),
            )
        providers[target] = item
    report = {
        "schema_version": 1,
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "architecture": detected,
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "providers": providers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
