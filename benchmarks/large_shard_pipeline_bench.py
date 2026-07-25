#!/usr/bin/env python3
"""Benchmark larger weight shards in a two-slot Llama layer pipeline.

The CPU arena represents 16 distinct decoder layers.  Each measured stage
copies a different contiguous slice into one of two GPU slots, then executes
the projection-linear stack for every layer in that slice.  This isolates the
effect of grouping multiple layers into a larger transfer while keeping total
weight traffic and total projection work constant.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from torch_llama32_1b_bench import (
    DTYPE,
    LAYER_BYTES_WITH_NORMS,
    LAYER_LINEAR_FLOPS_PER_ROW,
    LAYER_TOTAL_WEIGHT_ELEMENTS,
    LAYER_WEIGHT_ELEMENTS,
    MIB,
    LlamaLinearLayer,
    environment,
    nvidia_smi_snapshot,
    summary,
)


def parse_positive_ints(text, label):
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise SystemExit("{} must contain positive integers".format(label))
    if len(values) != len(set(values)):
        raise SystemExit("{} must not contain duplicates".format(label))
    return values


def make_events(stage_count, coordinator):
    events = {
        "start": torch.cuda.Event(enable_timing=True),
        "finish": torch.cuda.Event(enable_timing=True),
        "copy_begin": [
            torch.cuda.Event(enable_timing=True) for _ in range(stage_count)
        ],
        "ready": [
            torch.cuda.Event(enable_timing=True) for _ in range(stage_count)
        ],
        "compute_begin": [
            torch.cuda.Event(enable_timing=True) for _ in range(stage_count)
        ],
        "free": [
            torch.cuda.Event(enable_timing=True) for _ in range(stage_count)
        ],
    }
    all_events = [events["start"], events["finish"]]
    for name in ("copy_begin", "ready", "compute_begin", "free"):
        all_events.extend(events[name])
    for event in all_events:
        event.record(coordinator)
    events["finish"].synchronize()
    return events


def one_sample(
    x,
    host_groups,
    device_slots,
    slot_layers,
    copy_stream,
    compute_stream,
    coordinator,
    events,
    overlap,
):
    stage_count = len(host_groups)
    events["start"].record(coordinator)
    copy_stream.wait_event(events["start"])
    compute_stream.wait_event(events["start"])
    activation = x
    host_started = time.perf_counter_ns()
    copy_call_host_us = []
    copy_submission_host_us = []
    compute_submission_host_us = []

    for stage, host_group in enumerate(host_groups):
        slot = stage % 2
        if overlap and stage >= 2:
            copy_stream.wait_event(events["free"][stage - 2])
        elif not overlap and stage >= 1:
            copy_stream.wait_event(events["free"][stage - 1])

        copy_submission_started = time.perf_counter_ns()
        with torch.cuda.stream(copy_stream):
            events["copy_begin"][stage].record(copy_stream)
            copy_call_started = time.perf_counter_ns()
            device_slots[slot].copy_(host_group, non_blocking=True)
            copy_call_ended = time.perf_counter_ns()
            events["ready"][stage].record(copy_stream)
        copy_submission_ended = time.perf_counter_ns()
        copy_call_host_us.append(
            (copy_call_ended - copy_call_started) / 1000.0
        )
        copy_submission_host_us.append(
            (copy_submission_ended - copy_submission_started) / 1000.0
        )

        compute_submission_started = time.perf_counter_ns()
        compute_stream.wait_event(events["ready"][stage])
        with torch.cuda.stream(compute_stream):
            events["compute_begin"][stage].record(compute_stream)
            for layer in slot_layers[slot]:
                activation = layer(activation)[0]
            events["free"][stage].record(compute_stream)
        compute_submission_ended = time.perf_counter_ns()
        compute_submission_host_us.append(
            (compute_submission_ended - compute_submission_started) / 1000.0
        )

    coordinator.wait_event(events["free"][-1])
    events["finish"].record(coordinator)
    host_ended = time.perf_counter_ns()
    events["finish"].synchronize()

    stage_copy_ms = [
        events["copy_begin"][stage].elapsed_time(events["ready"][stage])
        for stage in range(stage_count)
    ]
    stage_compute_ms = [
        events["compute_begin"][stage].elapsed_time(events["free"][stage])
        for stage in range(stage_count)
    ]
    return {
        "total_ms": events["start"].elapsed_time(events["finish"]),
        "host_enqueue_us": (host_ended - host_started) / 1000.0,
        "copy_total_ms": sum(stage_copy_ms),
        "compute_total_ms": sum(stage_compute_ms),
        "copy_call_host_total_us": sum(copy_call_host_us),
        "copy_submission_host_total_us": sum(copy_submission_host_us),
        "compute_submission_host_total_us": sum(
            compute_submission_host_us
        ),
        "stage_copy_ms": stage_copy_ms,
        "stage_compute_ms": stage_compute_ms,
        "stage_completion_interval_ms": [
            events["free"][stage - 1].elapsed_time(events["free"][stage])
            for stage in range(1, stage_count)
        ],
    }


def summarize_mode(samples):
    intervals = [
        duration
        for sample in samples
        for duration in sample["stage_completion_interval_ms"]
    ]
    result = {
        "total_ms": summary([sample["total_ms"] for sample in samples]),
        "host_enqueue_us": summary(
            [sample["host_enqueue_us"] for sample in samples]
        ),
        "copy_total_ms": summary(
            [sample["copy_total_ms"] for sample in samples]
        ),
        "compute_total_ms": summary(
            [sample["compute_total_ms"] for sample in samples]
        ),
        "copy_call_host_total_us": summary(
            [sample["copy_call_host_total_us"] for sample in samples]
        ),
        "copy_submission_host_total_us": summary(
            [
                sample["copy_submission_host_total_us"]
                for sample in samples
            ]
        ),
        "compute_submission_host_total_us": summary(
            [
                sample["compute_submission_host_total_us"]
                for sample in samples
            ]
        ),
        "stage_copy_ms": summary(
            [
                duration
                for sample in samples
                for duration in sample["stage_copy_ms"]
            ]
        ),
        "stage_compute_ms": summary(
            [
                duration
                for sample in samples
                for duration in sample["stage_compute_ms"]
            ]
        ),
    }
    result["stage_completion_interval_ms"] = (
        summary(intervals) if intervals else None
    )
    return result


def build_slots(device, group_layers):
    group_elements = group_layers * LAYER_TOTAL_WEIGHT_ELEMENTS
    slots = [
        torch.empty(group_elements, dtype=DTYPE, device=device)
        for _ in range(2)
    ]
    layers = []
    for slot in slots:
        slot_views = []
        for layer_index in range(group_layers):
            begin = layer_index * LAYER_TOTAL_WEIGHT_ELEMENTS
            projection_slab = slot[begin : begin + LAYER_WEIGHT_ELEMENTS]
            slot_views.append(LlamaLinearLayer(device, slab=projection_slab))
        layers.append(slot_views)
    return slots, layers


def benchmark_group(
    device,
    host_arena,
    total_layers,
    group_layers,
    rows_per_call,
    warmup,
    repetitions,
):
    stage_count = total_layers // group_layers
    group_elements = group_layers * LAYER_TOTAL_WEIGHT_ELEMENTS
    host_groups = [
        host_arena[
            stage * group_elements : (stage + 1) * group_elements
        ]
        for stage in range(stage_count)
    ]
    device_slots, slot_layers = build_slots(device, group_layers)
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    coordinator = torch.cuda.current_stream(device)
    events = make_events(stage_count, coordinator)
    group_rows = []

    for rows in rows_per_call:
        x = torch.zeros((rows, 2048), dtype=DTYPE, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples = {"sequential": [], "overlap": []}

        with torch.inference_mode():
            for overlap in (False, True):
                one_sample(
                    x,
                    host_groups,
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
                        host_groups,
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
                            host_groups,
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
            name: summarize_mode(mode_samples)
            for name, mode_samples in samples.items()
        }
        sequential_ms = modes["sequential"]["total_ms"]["median"]
        overlap_ms = modes["overlap"]["total_ms"]["median"]
        stage_copy_ms = modes["sequential"]["stage_copy_ms"]["median"]
        stage_compute_ms = modes["sequential"]["stage_compute_ms"]["median"]
        component_sequential_ms = stage_count * (
            stage_copy_ms + stage_compute_ms
        )
        component_ideal_overlap_ms = (
            stage_count * max(stage_copy_ms, stage_compute_ms)
            + min(stage_copy_ms, stage_compute_ms)
        )
        total_weight_bytes = total_layers * LAYER_BYTES_WITH_NORMS
        total_flops = (
            total_layers * LAYER_LINEAR_FLOPS_PER_ROW * rows
        )
        sequential_copy_total_ms = modes["sequential"]["copy_total_ms"][
            "median"
        ]
        sequential_compute_total_ms = modes["sequential"][
            "compute_total_ms"
        ]["median"]
        row = {
            "rows_M": rows,
            "modes": modes,
            "speedup_vs_sequential": sequential_ms / overlap_ms,
            "saved_ms_vs_sequential": sequential_ms - overlap_ms,
            "ideal_no_interference_speedup_from_sequential_components": (
                component_sequential_ms / component_ideal_overlap_ms
            ),
            "overlap_efficiency_vs_component_ideal": (
                (sequential_ms / overlap_ms)
                / (component_sequential_ms / component_ideal_overlap_ms)
            ),
            "pipeline_passes_per_second": 1000.0 / overlap_ms,
            "rows_per_second": rows * 1000.0 / overlap_ms,
            "end_to_end_weight_stream_GBps": (
                total_weight_bytes / (overlap_ms * 1e6)
            ),
            "end_to_end_projection_TFLOPs": (
                total_flops / (overlap_ms * 1e9)
            ),
            "sequential_component_H2D_GBps": (
                total_weight_bytes / (sequential_copy_total_ms * 1e6)
            ),
            "sequential_component_projection_TFLOPs": (
                total_flops / (sequential_compute_total_ms * 1e9)
            ),
            "stage_copy_effective_GBps_sequential": (
                group_layers
                * LAYER_BYTES_WITH_NORMS
                / (stage_copy_ms * 1e6)
            ),
            "copy_slowdown_during_overlap": (
                modes["overlap"]["stage_copy_ms"]["median"] / stage_copy_ms
            ),
            "compute_slowdown_during_overlap": (
                modes["overlap"]["stage_compute_ms"]["median"]
                / stage_compute_ms
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        group_rows.append(row)
        print(
            "g={:2d}, M={:5d}, stages={:2d}: seq {:9.3f} ms, "
            "overlap {:9.3f} ms, speedup {:.3f}x, {:.2f} passes/s".format(
                group_layers,
                rows,
                stage_count,
                sequential_ms,
                overlap_ms,
                row["speedup_vs_sequential"],
                row["pipeline_passes_per_second"],
            ),
            flush=True,
        )
        del x

    torch.cuda.synchronize(device)
    del events, slot_layers, device_slots, host_groups
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "group_layers": group_layers,
        "stage_count": stage_count,
        "slot_bytes": group_layers * LAYER_BYTES_WITH_NORMS,
        "two_slot_bytes": 2 * group_layers * LAYER_BYTES_WITH_NORMS,
        "rows": group_rows,
    }


def add_relative_throughput(groups):
    baseline = {}
    for group in groups:
        if group["group_layers"] == 1:
            baseline = {
                row["rows_M"]: row["modes"]["overlap"]["total_ms"]["median"]
                for row in group["rows"]
            }
            break
    if not baseline:
        return
    for group in groups:
        for row in group["rows"]:
            row["overlap_throughput_relative_to_g1"] = (
                baseline[row["rows_M"]]
                / row["modes"]["overlap"]["total_ms"]["median"]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--total-layers", type=int, default=16)
    parser.add_argument("--group-layers", default="1,2,4,8")
    parser.add_argument("--rows", default="1,512,2048,4096")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/large_shard_pipeline.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise SystemExit("CUDA device {} does not exist".format(args.device))
    if args.total_layers < 2:
        raise SystemExit("--total-layers must be at least 2")
    if args.warmup < 0 or args.repetitions <= 0:
        raise SystemExit("warmup must be non-negative and repetitions positive")

    groups = parse_positive_ints(args.group_layers, "--group-layers")
    rows_per_call = parse_positive_ints(args.rows, "--rows")
    for group_layers in groups:
        if args.total_layers % group_layers != 0:
            raise SystemExit(
                "{} layers is not divisible by group size {}".format(
                    args.total_layers, group_layers
                )
            )

    torch.cuda.set_device(args.device)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("selected GPU does not support BF16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    total_elements = args.total_layers * LAYER_TOTAL_WEIGHT_ELEMENTS
    print(
        "Allocating full {}-layer pinned arena: {:.3f} MiB".format(
            args.total_layers,
            args.total_layers * LAYER_BYTES_WITH_NORMS / MIB,
        ),
        flush=True,
    )
    host_arena = torch.empty(total_elements, dtype=DTYPE, pin_memory=True)
    host_arena.zero_()

    result = {
        "schema_version": 1,
        "benchmark": "llama32_1b_large_shard_two_slot_pipeline",
        "environment": environment(args.device),
        "settings": {
            "device": args.device,
            "total_layers": args.total_layers,
            "group_layers_order": groups,
            "rows": rows_per_call,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "output": str(args.output),
        },
        "model": {
            "id": "meta-llama/Llama-3.2-1B",
            "synthetic_zero_weights": True,
            "dtype": str(DTYPE),
            "compute_scope": "decoder_projection_linear_only",
            "layer_bytes_with_norms": LAYER_BYTES_WITH_NORMS,
            "total_streamed_weight_bytes": (
                args.total_layers * LAYER_BYTES_WITH_NORMS
            ),
        },
        "cpu_weight_arena": {
            "pinned": True,
            "bytes": args.total_layers * LAYER_BYTES_WITH_NORMS,
            "distinct_layer_slices": True,
            "allocation_and_initialization_excluded_from_timing": True,
        },
        "pipeline": {
            "device_weight_slots": 2,
            "dependency": (
                "overlap copy(i>=2) waits free(i-2); sequential copy(i>=1) "
                "waits free(i-1); compute(i) waits ready(i)"
            ),
            "sample_sync": (
                "one finish-event synchronization after the full 16-layer pass"
            ),
            "groups": [],
        },
    }

    for group_layers in groups:
        result["pipeline"]["groups"].append(
            benchmark_group(
                args.device,
                host_arena,
                args.total_layers,
                group_layers,
                rows_per_call,
                args.warmup,
                args.repetitions,
            )
        )
    add_relative_throughput(result["pipeline"]["groups"])
    torch.cuda.synchronize(args.device)
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
