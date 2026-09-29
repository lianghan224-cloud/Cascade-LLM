#!/usr/bin/env python3
"""Small-batch MoE backend and dispatch benchmark (synthetic only)."""

import argparse
import json
import statistics
import time

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from layer_streaming.moe import (
    ExpertDispatcher,
    ExpertWeights,
    GroupedExpertBackend,
    MoEConfig,
    MoEFFN,
    NaiveExpertBackend,
    Router,
    StackedExpertWeights,
)


def build_runtime(block, config, grouped):
    experts = [
        ExpertWeights(
            item.gate_proj.weight, item.up_proj.weight, item.down_proj.weight
        )
        for item in block.experts
    ]
    backend = GroupedExpertBackend() if grouped else NaiveExpertBackend()
    weights = (
        StackedExpertWeights.from_experts(experts)
        if grouped
        else {index: value for index, value in enumerate(experts)}
    )
    moe = MoEConfig(
        num_experts=config.num_experts,
        experts_per_token=config.num_experts_per_tok,
        expert_intermediate_size=config.intermediate_size,
        router_dtype="float32",
        normalize_topk=config.norm_topk_prob,
    )
    return MoEFFN(
        Router(moe, block.gate.weight),
        ExpertDispatcher(moe.num_experts),
        backend,
        weights,
    )


def benchmark(runtime, hidden, iterations, device):
    for _ in range(5):
        runtime(hidden)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            runtime(hidden)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iterations
    started = time.perf_counter()
    for _ in range(iterations):
        runtime(hidden)
    return (time.perf_counter() - started) * 1000.0 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device(args.device)
    config = OlmoeConfig(
        vocab_size=67,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        attention_bias=False,
        tie_word_embeddings=False,
        norm_topk_prob=False,
        torch_dtype="float32",
    )
    block = OlmoeForCausalLM(config).model.layers[0].mlp.to(device).eval()
    naive = build_runtime(block, config, False)
    grouped = build_runtime(block, config, True)
    rows = []
    with torch.inference_mode():
        for tokens in (1, 2, 4, 8, 32):
            hidden = torch.randn(tokens, config.hidden_size, device=device)
            expected = naive(hidden)[0]
            actual = grouped(hidden)[0]
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            naive_ms = benchmark(naive, hidden, args.iterations, device)
            grouped_ms = benchmark(grouped, hidden, args.iterations, device)
            rows.append(
                {
                    "iteration_tokens": tokens,
                    "naive_ms": naive_ms,
                    "grouped_ms": grouped_ms,
                    "grouped_speedup": naive_ms / grouped_ms,
                }
            )
    result = {
        "qualification": "SYNTHETIC_ONLY",
        "device": str(device),
        "iterations": args.iterations,
        "rows": rows,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
