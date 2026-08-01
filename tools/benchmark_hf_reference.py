#!/usr/bin/env python3
"""Small Hugging Face baseline for real-checkpoint Cascade comparisons."""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM


def parse_ids(value):
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("token list cannot be empty")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure a minimal Hugging Face eager baseline with the same "
            "prefill/decode split used by Cascade-LLM."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device-map",
        choices=("single", "balanced"),
        default="balanced",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--input-ids",
        type=parse_ids,
        default=[128000, 4, 5, 6, 7, 8, 9, 10],
    )
    parser.add_argument("--decode-ids", type=parse_ids, default=[4, 5])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def checkpoint_bytes(checkpoint):
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        size = index.get("metadata", {}).get("total_size")
        if size is not None:
            return int(size)
        files = set(index.get("weight_map", {}).values())
        return sum((checkpoint / name).stat().st_size for name in files)
    return (checkpoint / "model.safetensors").stat().st_size


def synchronize_all():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def memory_snapshot():
    return [
        {
            "device": index,
            "name": torch.cuda.get_device_name(index),
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(index)
            ),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }
        for index in range(torch.cuda.device_count())
    ]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    config = AutoConfig.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    model_bytes = checkpoint_bytes(args.checkpoint)
    requested_device = torch.device(args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(requested_device)
    single_supported = model_bytes < free_bytes
    report = {
        "schema_version": 1,
        "framework": "huggingface_transformers",
        "checkpoint": str(args.checkpoint.resolve()),
        "device_map": args.device_map,
        "input_ids": args.input_ids,
        "decode_ids": args.decode_ids,
        "weight_dtype": str(config.torch_dtype or "unknown"),
        "checkpoint_bytes": model_bytes,
        "single_gpu_preflight": {
            "supported": single_supported,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "reason": (
                None
                if single_supported
                else "checkpoint weights alone exceed current single-GPU free memory"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    if args.preflight_only or (
        args.device_map == "single" and not single_supported
    ):
        report["status"] = (
            "preflight_only" if args.preflight_only else "unsupported"
        )
    else:
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=(
                "balanced"
                if args.device_map == "balanced"
                else {"": str(requested_device)}
            ),
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        synchronize_all()
        load_seconds = time.perf_counter() - load_started
        input_device = model.model.embed_tokens.weight.device
        prompt = torch.tensor(
            [args.input_ids], dtype=torch.long, device=input_device
        )
        decode_tokens = [
            torch.tensor([[token]], dtype=torch.long, device=input_device)
            for token in args.decode_ids
        ]
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
        with torch.inference_mode():
            started = time.perf_counter()
            output = model(prompt, use_cache=True)
            synchronize_all()
            ttft_ms = (time.perf_counter() - started) * 1000.0
            cache = output.past_key_values
            topk = torch.topk(
                output.logits[:, -1, :].float(),
                min(args.top_k, config.vocab_size),
                dim=-1,
            ).indices.cpu().tolist()
            decode_latencies = []
            for token in decode_tokens:
                started = time.perf_counter()
                output = model(
                    token, past_key_values=cache, use_cache=True
                )
                synchronize_all()
                decode_latencies.append(
                    (time.perf_counter() - started) * 1000.0
                )
                cache = output.past_key_values
        report.update(
            {
                "status": "ok",
                "load_time_seconds": load_seconds,
                "ttft_ms": ttft_ms,
                "decode_latencies_ms": decode_latencies,
                "decode_ms_per_token": (
                    sum(decode_latencies) / len(decode_latencies)
                    if decode_latencies
                    else 0.0
                ),
                "topk_ids": topk,
                "hf_device_map": {
                    str(key): str(value)
                    for key, value in getattr(
                        model, "hf_device_map", {}
                    ).items()
                },
                "gpu_memory": memory_snapshot(),
                "notes": [
                    "eager Hugging Face baseline; no quantization",
                    "balanced mode uses every visible GPU and is not a single-GPU throughput comparison",
                ],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
