#!/usr/bin/env python3
"""Run local Llama-3.1-8B weights through the streaming runtime."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    DoubleBufferRuntime,
    Llama31DecodeExecutor,
    ResidentDeviceArena,
    VocabStreamingRuntime,
    build_llama31_8b_plan,
    create_weight_store,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="full_pinned",
    )
    parser.add_argument(
        "--granularity",
        choices=("matrix", "matrix_group", "layer"),
        default="matrix_group",
    )
    parser.add_argument(
        "--vocab-mode",
        choices=("resident", "streamed"),
        default="streamed",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    config = AutoConfig.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    plan = build_llama31_8b_plan(
        args.granularity,
        tie_word_embeddings=bool(config.tie_word_embeddings),
        stream_vocab=(args.vocab_mode == "streamed"),
    )
    store = create_weight_store(plan, args.weight_store)
    load_started = time.perf_counter()
    store.load_checkpoint(args.checkpoint)
    load_seconds = time.perf_counter() - load_started

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    resident = ResidentDeviceArena(plan, store, device)
    runtime = DoubleBufferRuntime(plan, store, resident, device)
    vocab_runtime = None
    if plan.vocab is not None:
        vocab_runtime = VocabStreamingRuntime(
            plan,
            store,
            runtime,
        )
    executor = Llama31DecodeExecutor(
        config,
        resident,
        vocab_runtime=vocab_runtime,
        top_k=args.top_k,
    )
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded.input_ids.to(device)
    generated = []
    token_latencies = []

    with torch.inference_mode():
        state = executor.begin(input_ids)
        state = runtime.run(executor, state)
        state = executor.finish(state)
        next_token = state.topk_indices[..., 0]
        generated.append(next_token)

        for _ in range(args.max_new_tokens - 1):
            started = time.perf_counter()
            state = executor.begin(next_token)
            state = runtime.run(executor, state)
            state = executor.finish(state)
            next_token = state.topk_indices[..., 0]
            torch.cuda.synchronize(device)
            token_latencies.append(
                (time.perf_counter() - started) * 1000.0
            )
            generated.append(next_token)

    generated_ids = torch.cat(generated, dim=-1).cpu()
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    result = {
        "model_id": plan.model_id,
        "checkpoint": str(args.checkpoint),
        "weight_store": args.weight_store,
        "granularity": args.granularity,
        "vocab_mode": args.vocab_mode,
        "top_k": args.top_k,
        "prompt_tokens": int(input_ids.numel()),
        "generated_tokens": args.max_new_tokens,
        "generated_text": text,
        "checkpoint_load_seconds": load_seconds,
        "decode_token_latencies_ms": token_latencies,
        "runtime": runtime.stats.as_dict(),
        "cpu_pinned_bytes": store.pinned_cpu_bytes,
        "vocab_extra_pinned_cpu_bytes": (
            vocab_runtime.extra_pinned_cpu_bytes
            if vocab_runtime is not None
            else 0
        ),
        "plan": {
            "host_arena_bytes": plan.host_arena_bytes,
            "stream_bytes_per_token": plan.stream_bytes_per_token,
        },
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    store.close()


if __name__ == "__main__":
    main()
