#!/usr/bin/env python3
"""Save reproducible reference or streaming logits for real Llama-3.1-8B."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    DoubleBufferRuntime,
    Llama31DecodeExecutor,
    ResidentDeviceArena,
    build_llama31_8b_plan,
    create_weight_store,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("reference", "streaming"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="pinned_staging",
    )
    parser.add_argument(
        "--granularity",
        choices=("matrix", "layer"),
        default="matrix",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    return parser.parse_args()


def memory_snapshot():
    process = psutil.Process()
    memory = process.memory_info()
    io = process.io_counters()
    return {
        "rss_bytes": memory.rss,
        "vms_bytes": memory.vms,
        "read_bytes": io.read_bytes,
        "write_bytes": io.write_bytes,
    }


def topk_summary(logits, count=10):
    probabilities = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(logits, k=count, dim=-1)
    selected_probabilities = probabilities.gather(-1, indices)
    return {
        "token_ids": indices.tolist(),
        "logits": values.tolist(),
        "probabilities": selected_probabilities.tolist(),
    }


def run_reference(args, tokenizer, input_ids):
    torch.set_num_threads(args.cpu_threads)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    after_load = memory_snapshot()

    generated = []
    logits_rows = []
    latencies = []
    current = input_ids
    cache = None
    with torch.inference_mode():
        for _ in range(args.steps):
            started = time.perf_counter()
            outputs = model(
                input_ids=current,
                past_key_values=cache,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :].float().cpu()
            latencies.append((time.perf_counter() - started) * 1000.0)
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            logits_rows.append(logits)
            current = next_token
            cache = outputs.past_key_values
    return {
        "load_seconds": load_seconds,
        "after_load": after_load,
        "step_latencies_ms": latencies,
        "generated": generated,
        "logits": logits_rows,
        "runtime": None,
        "cpu_pinned_bytes": 0,
        "cuda_peak_allocated_bytes": 0,
    }


def run_streaming(args, config, input_ids):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    plan = build_llama31_8b_plan(
        args.granularity,
        tie_word_embeddings=bool(config.tie_word_embeddings),
    )
    store = create_weight_store(plan, args.weight_store)
    load_started = time.perf_counter()
    store.load_checkpoint(args.checkpoint)
    load_seconds = time.perf_counter() - load_started
    after_load = memory_snapshot()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident = ResidentDeviceArena(plan, store, device)
    runtime = DoubleBufferRuntime(plan, store, resident, device)
    executor = Llama31DecodeExecutor(config, resident)

    generated = []
    logits_rows = []
    latencies = []
    current = input_ids.to(device)
    with torch.inference_mode():
        for _ in range(args.steps):
            started = time.perf_counter()
            state = executor.begin(current)
            state = runtime.run(executor, state)
            state = executor.finish(state)
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)
            logits = state.logits[:, -1, :].float().cpu()
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            logits_rows.append(logits)
            current = next_token.to(device)
    result = {
        "load_seconds": load_seconds,
        "after_load": after_load,
        "step_latencies_ms": latencies,
        "generated": generated,
        "logits": logits_rows,
        "runtime": runtime.stats.as_dict(),
        "cpu_pinned_bytes": store.pinned_cpu_bytes,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    store.close()
    return result


def main():
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    config = AutoConfig.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    before_load = memory_snapshot()
    if args.mode == "reference":
        result = run_reference(args, tokenizer, encoded.input_ids)
    else:
        result = run_streaming(args, config, encoded.input_ids)
    after_run = memory_snapshot()

    generated_ids = torch.cat(result.pop("generated"), dim=-1)
    logits = torch.cat(result.pop("logits"), dim=0).contiguous()
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"logits": logits}, str(args.logits_output))
    topk = [topk_summary(row) for row in logits]
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "input_ids": encoded.input_ids.tolist(),
        "steps": args.steps,
        "generated_token_ids": generated_ids.tolist(),
        "generated_text": tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ),
        "topk": topk,
        "logits_file": str(args.logits_output),
        "before_load": before_load,
        "after_run": after_run,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cpu_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    report.update({key: value for key, value in result.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
