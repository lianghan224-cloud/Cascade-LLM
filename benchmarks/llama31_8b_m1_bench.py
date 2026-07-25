#!/usr/bin/env python3
"""M=1 component accounting for exact-shape Llama-3.1-8B layers."""

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from torch_llama32_1b_bench import (
    DTYPE,
    environment,
    nvidia_smi_snapshot,
    summary,
)


MIB = 1024 * 1024
GIB = 1024 * MIB
DTYPE_BYTES = 2
TOTAL_LAYERS = 32
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
KV_SIZE = 1024
LLAMA31_8B_SHAPES = (
    ("q_proj", (HIDDEN_SIZE, HIDDEN_SIZE)),
    ("k_proj", (KV_SIZE, HIDDEN_SIZE)),
    ("v_proj", (KV_SIZE, HIDDEN_SIZE)),
    ("o_proj", (HIDDEN_SIZE, HIDDEN_SIZE)),
    ("gate_proj", (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
    ("up_proj", (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
    ("down_proj", (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
)
NORM_ELEMENTS = 2 * HIDDEN_SIZE
LAYER_WEIGHT_ELEMENTS = sum(
    math.prod(shape) for _, shape in LLAMA31_8B_SHAPES
)
LAYER_TOTAL_WEIGHT_ELEMENTS = LAYER_WEIGHT_ELEMENTS + NORM_ELEMENTS
LAYER_BYTES = LAYER_TOTAL_WEIGHT_ELEMENTS * DTYPE_BYTES
LAYER_FLOPS_PER_ROW = 2 * LAYER_WEIGHT_ELEMENTS


class Llama31LinearLayer:
    def __init__(self, device, slab):
        if slab.numel() != LAYER_WEIGHT_ELEMENTS or slab.dtype != DTYPE:
            raise ValueError("invalid Llama-3.1-8B projection slab")
        self.device = device
        self.weights = {}
        offset = 0
        for name, shape in LLAMA31_8B_SHAPES:
            elements = math.prod(shape)
            self.weights[name] = slab[offset : offset + elements].view(shape)
            offset += elements

    def __call__(self, x):
        q = F.linear(x, self.weights["q_proj"])
        k = F.linear(x, self.weights["k_proj"])
        v = F.linear(x, self.weights["v_proj"])
        attention_output = F.linear(q, self.weights["o_proj"])
        gate = F.linear(x, self.weights["gate_proj"])
        up = F.linear(x, self.weights["up_proj"])
        mlp_output = F.linear(
            F.silu(gate) * up, self.weights["down_proj"]
        )
        return attention_output + mlp_output, k, v


def parse_groups(text):
    groups = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not groups or any(group <= 0 for group in groups):
        raise SystemExit("--group-layers must contain positive integers")
    if len(groups) != len(set(groups)):
        raise SystemExit("--group-layers must not contain duplicates")
    if any(TOTAL_LAYERS % group != 0 for group in groups):
        raise SystemExit("every group size must divide 32 layers")
    return groups


def make_events(stage_count, coordinator):
    events = {
        "start": torch.cuda.Event(enable_timing=True),
        "finish": torch.cuda.Event(enable_timing=True),
    }
    for name in ("copy_begin", "ready", "compute_begin", "free"):
        events[name] = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(stage_count)
        ]
    all_events = [events["start"], events["finish"]]
    for name in ("copy_begin", "ready", "compute_begin", "free"):
        all_events.extend(events[name])
    for event in all_events:
        event.record(coordinator)
    events["finish"].synchronize()
    return events


def build_slots(device, group_layers):
    group_elements = group_layers * LAYER_TOTAL_WEIGHT_ELEMENTS
    slots = [
        torch.empty(group_elements, dtype=DTYPE, device=device)
        for _ in range(2)
    ]
    layers = []
    for slot in slots:
        group = []
        for layer_index in range(group_layers):
            begin = layer_index * LAYER_TOTAL_WEIGHT_ELEMENTS
            slab = slot[begin : begin + LAYER_WEIGHT_ELEMENTS]
            group.append(Llama31LinearLayer(device, slab))
        layers.append(group)
    return slots, layers


def one_sample(
    x,
    host_group,
    device_slots,
    slot_layers,
    copy_stream,
    compute_stream,
    coordinator,
    events,
    overlap,
):
    stage_count = len(events["ready"])
    events["start"].record(coordinator)
    copy_stream.wait_event(events["start"])
    compute_stream.wait_event(events["start"])
    activation = x
    copy_call_us = []
    copy_submission_us = []
    compute_submission_us = []
    host_started = time.perf_counter_ns()

    for stage in range(stage_count):
        slot = stage % 2
        if overlap and stage >= 2:
            copy_stream.wait_event(events["free"][stage - 2])
        elif not overlap and stage >= 1:
            copy_stream.wait_event(events["free"][stage - 1])

        submission_started = time.perf_counter_ns()
        with torch.cuda.stream(copy_stream):
            events["copy_begin"][stage].record(copy_stream)
            call_started = time.perf_counter_ns()
            device_slots[slot].copy_(host_group, non_blocking=True)
            call_ended = time.perf_counter_ns()
            events["ready"][stage].record(copy_stream)
        submission_ended = time.perf_counter_ns()
        copy_call_us.append((call_ended - call_started) / 1000.0)
        copy_submission_us.append(
            (submission_ended - submission_started) / 1000.0
        )

        submission_started = time.perf_counter_ns()
        compute_stream.wait_event(events["ready"][stage])
        with torch.cuda.stream(compute_stream):
            events["compute_begin"][stage].record(compute_stream)
            for layer in slot_layers[slot]:
                activation = layer(activation)[0]
            events["free"][stage].record(compute_stream)
        submission_ended = time.perf_counter_ns()
        compute_submission_us.append(
            (submission_ended - submission_started) / 1000.0
        )

    coordinator.wait_event(events["free"][-1])
    events["finish"].record(coordinator)
    host_ended = time.perf_counter_ns()
    events["finish"].synchronize()
    copy_durations = [
        events["copy_begin"][stage].elapsed_time(events["ready"][stage])
        for stage in range(stage_count)
    ]
    compute_durations = [
        events["compute_begin"][stage].elapsed_time(events["free"][stage])
        for stage in range(stage_count)
    ]
    return {
        "total_ms": events["start"].elapsed_time(events["finish"]),
        "copy_total_ms": sum(copy_durations),
        "compute_total_ms": sum(compute_durations),
        "copy_call_host_total_us": sum(copy_call_us),
        "copy_submission_host_total_us": sum(copy_submission_us),
        "compute_submission_host_total_us": sum(compute_submission_us),
        "full_host_enqueue_us": (host_ended - host_started) / 1000.0,
    }


def summarize_samples(samples):
    return {
        name: summary([sample[name] for sample in samples])
        for name in samples[0]
    }


def benchmark_group(device, group_layers, warmup, repetitions):
    group_elements = group_layers * LAYER_TOTAL_WEIGHT_ELEMENTS
    stage_count = TOTAL_LAYERS // group_layers
    host_group = torch.empty(
        group_elements, dtype=DTYPE, pin_memory=True
    )
    host_group.zero_()
    device_slots, slot_layers = build_slots(device, group_layers)
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    coordinator = torch.cuda.current_stream(device)
    events = make_events(stage_count, coordinator)
    x = torch.zeros((1, HIDDEN_SIZE), dtype=DTYPE, device=device)
    torch.cuda.synchronize(device)
    samples = {"sequential": [], "overlap": []}

    with torch.inference_mode():
        for overlap in (False, True):
            one_sample(
                x,
                host_group,
                device_slots,
                slot_layers,
                copy_stream,
                compute_stream,
                coordinator,
                events,
                overlap,
            )
        for _ in range(warmup):
            for overlap in (False, True):
                one_sample(
                    x,
                    host_group,
                    device_slots,
                    slot_layers,
                    copy_stream,
                    compute_stream,
                    coordinator,
                    events,
                    overlap,
                )
        for repetition in range(repetitions):
            order = (False, True) if repetition % 2 == 0 else (True, False)
            for overlap in order:
                name = "overlap" if overlap else "sequential"
                samples[name].append(
                    one_sample(
                        x,
                        host_group,
                        device_slots,
                        slot_layers,
                        copy_stream,
                        compute_stream,
                        coordinator,
                        events,
                        overlap,
                    )
                )

    modes = {
        name: summarize_samples(mode_samples)
        for name, mode_samples in samples.items()
    }
    sequential_ms = modes["sequential"]["total_ms"]["median"]
    overlap_ms = modes["overlap"]["total_ms"]["median"]
    copy_ms = modes["sequential"]["copy_total_ms"]["median"]
    compute_ms = modes["sequential"]["compute_total_ms"]["median"]
    total_weight_bytes = TOTAL_LAYERS * LAYER_BYTES
    total_flops = TOTAL_LAYERS * LAYER_FLOPS_PER_ROW
    result = {
        "status": "measured",
        "group_layers": group_layers,
        "stage_count": stage_count,
        "shard_bytes": group_layers * LAYER_BYTES,
        "two_slot_bytes": 2 * group_layers * LAYER_BYTES,
        "source_group_reused_for_all_stages": True,
        "modes": modes,
        "metrics": {
            "h2d_total_GBps": total_weight_bytes / (copy_ms * 1e6),
            "compute_total_TFLOPs": total_flops / (compute_ms * 1e9),
            "speedup_vs_sequential": sequential_ms / overlap_ms,
            "saved_by_overlap_ms": sequential_ms - overlap_ms,
            "h2d_fraction_of_sequential": copy_ms / sequential_ms,
            "compute_fraction_of_sequential": compute_ms / sequential_ms,
        },
    }
    print(
        "g={:2d}, stages={:2d}: H2D {:9.3f} ms, compute {:7.3f} ms, "
        "serial {:9.3f} ms, overlap {:9.3f} ms, {:.3f}x".format(
            group_layers,
            stage_count,
            copy_ms,
            compute_ms,
            sequential_ms,
            overlap_ms,
            result["metrics"]["speedup_vs_sequential"],
        ),
        flush=True,
    )

    torch.cuda.synchronize(device)
    del x, events, slot_layers, device_slots, host_group
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--group-layers", default="1,2,4,8,16,32")
    parser.add_argument("--arena-gib", type=float, default=8.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/llama31_8b_m1.json"),
    )
    args = parser.parse_args()
    groups = parse_groups(args.group_layers)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise SystemExit("invalid CUDA device")
    if args.warmup < 0 or args.repetitions <= 0:
        raise SystemExit("invalid warmup/repetitions")

    torch.cuda.set_device(args.device)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("selected GPU does not support BF16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    arena_bytes = int(args.arena_gib * GIB)
    device_bytes = torch.cuda.get_device_properties(args.device).total_memory
    result = {
        "schema_version": 1,
        "benchmark": "llama31_8b_M1_shard_accounting",
        "environment": environment(args.device),
        "settings": {
            "device": args.device,
            "rows_M": 1,
            "group_layers_order": groups,
            "arena_bytes": arena_bytes,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "output": str(args.output),
        },
        "model": {
            "id": "meta-llama/Llama-3.1-8B",
            "dtype": str(DTYPE),
            "synthetic_zero_weights": True,
            "compute_scope": "decoder_projection_linear_only",
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_hidden_layers": TOTAL_LAYERS,
            "num_key_value_heads": 8,
            "layer_linear_weight_elements": LAYER_WEIGHT_ELEMENTS,
            "layer_bytes_with_norms": LAYER_BYTES,
            "decoder_stream_bytes": TOTAL_LAYERS * LAYER_BYTES,
            "shapes": [
                {"name": name, "shape": list(shape)}
                for name, shape in LLAMA31_8B_SHAPES
            ],
        },
        "groups": [],
    }

    for group_layers in groups:
        shard_bytes = group_layers * LAYER_BYTES
        two_slot_bytes = 2 * shard_bytes
        if two_slot_bytes > arena_bytes:
            result["groups"].append(
                {
                    "status": "infeasible_for_arena",
                    "group_layers": group_layers,
                    "stage_count": TOTAL_LAYERS // group_layers,
                    "shard_bytes": shard_bytes,
                    "two_slot_bytes": two_slot_bytes,
                    "arena_bytes": arena_bytes,
                    "reason": "two weight slots exceed configured arena",
                }
            )
            print(
                "g={:2d}: skipped, two slots {:.3f} GiB exceed {:.3f} "
                "GiB arena".format(
                    group_layers,
                    two_slot_bytes / GIB,
                    arena_bytes / GIB,
                ),
                flush=True,
            )
            continue
        if two_slot_bytes >= device_bytes:
            result["groups"].append(
                {
                    "status": "infeasible_for_device",
                    "group_layers": group_layers,
                    "stage_count": TOTAL_LAYERS // group_layers,
                    "shard_bytes": shard_bytes,
                    "two_slot_bytes": two_slot_bytes,
                    "device_bytes": device_bytes,
                    "reason": "two weight slots exceed physical GPU memory",
                }
            )
            continue
        result["groups"].append(
            benchmark_group(
                args.device,
                group_layers,
                args.warmup,
                args.repetitions,
            )
        )

    result["environment"]["nvidia_smi_after_benchmark"] = (
        nvidia_smi_snapshot()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    print("Wrote {}".format(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
