#!/usr/bin/env python3
"""Calibrate H2D, Llama-3.2-1B projection compute, and copy/compute overlap.

This benchmark uses the exact BF16 projection shapes of one
meta-llama/Llama-3.2-1B decoder layer.  The compute microbenchmark is
projection-linear-only; it intentionally omits attention, RoPE, RMSNorm, KV
cache work, and residuals.  It does not need model weights.
"""

import argparse
import ctypes
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


MIB = 1024 * 1024
GIB = 1024 * MIB
DTYPE = torch.bfloat16
DTYPE_BYTES = 2

# torch.nn.functional.linear stores weights as [out_features, in_features].
LLAMA_SHAPES = (
    ("q_proj", (2048, 2048)),
    ("k_proj", (512, 2048)),
    ("v_proj", (512, 2048)),
    ("o_proj", (2048, 2048)),
    ("gate_proj", (8192, 2048)),
    ("up_proj", (8192, 2048)),
    ("down_proj", (2048, 8192)),
)
NORM_ELEMENTS = 2 * 2048
LAYER_WEIGHT_ELEMENTS = sum(math.prod(shape) for _, shape in LLAMA_SHAPES)
LAYER_TOTAL_WEIGHT_ELEMENTS = LAYER_WEIGHT_ELEMENTS + NORM_ELEMENTS
LAYER_LINEAR_WEIGHT_BYTES = LAYER_WEIGHT_ELEMENTS * DTYPE_BYTES
LAYER_BYTES_WITH_NORMS = LAYER_TOTAL_WEIGHT_ELEMENTS * DTYPE_BYTES
LAYER_LINEAR_FLOPS_PER_ROW = 2 * LAYER_WEIGHT_ELEMENTS
CPU_STAGING_RING_MIN_BYTES = 160 * MIB


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summary(values):
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
        "samples": len(values),
    }


def linear_fit(xs, ys):
    """Return intercept and slope for y = intercept + slope*x."""
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = slope / denominator
    intercept = y_mean - slope * x_mean
    residual = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)
    )
    total = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    return intercept, slope, r_squared


def run_command(command):
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True, timeout=10
        ).strip()
    except Exception as exc:
        return "unavailable: {}".format(exc)


def nvidia_smi_snapshot():
    return run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,name,driver_version,pstate,"
            "pcie.link.gen.current,pcie.link.gen.max,"
            "pcie.link.width.current,pcie.link.width.max,"
            "clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader",
        ]
    )


def environment(device):
    properties = torch.cuda.get_device_properties(device)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device_index": device,
        "device_name": properties.name,
        "compute_capability": "{}.{}".format(properties.major, properties.minor),
        "multiprocessors": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "async_engine_count": getattr(properties, "async_engine_count", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_before_warmup": nvidia_smi_snapshot(),
    }


def event_measure(operation, warmup, repetitions):
    # Always perform one untimed API/shape prime, even with --warmup=0.
    operation()
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    elapsed_ms = []
    host_enqueue_us = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        host_start_ns = time.perf_counter_ns()
        operation()
        host_end_ns = time.perf_counter_ns()
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))
        host_enqueue_us.append((host_end_ns - host_start_ns) / 1000.0)
    return elapsed_ms, host_enqueue_us


def h2d_sweep(device, sizes, warmup, repetitions):
    results = []
    for pinned in (True, False):
        for nbytes in sizes:
            allocation_bytes = max(1, nbytes)
            host = torch.empty(
                allocation_bytes, dtype=torch.uint8, pin_memory=pinned
            )
            destination = torch.empty(
                allocation_bytes, dtype=torch.uint8, device=device
            )
            host.zero_()
            source = host[:nbytes]
            target = destination[:nbytes]

            def copy():
                target.copy_(source, non_blocking=True)

            elapsed_ms, host_us = event_measure(copy, warmup, repetitions)
            median_ms = statistics.median(elapsed_ms)
            gbps = (
                nbytes / (median_ms / 1000.0) / 1e9
                if nbytes and median_ms > 0
                else 0.0
            )
            row = {
                "pinned": pinned,
                "bytes": nbytes,
                "gpu_elapsed_ms": summary(elapsed_ms),
                "host_call_us": summary(host_us),
                "effective_GBps": gbps,
                "measurement_scope": (
                    "pytorch_eager_copy_effective"
                    if pinned
                    else "pageable_effective_end_to_end"
                ),
            }
            results.append(row)
            print(
                "H2D {:8s} {:9.3f} MiB: {:9.4f} ms, {:7.3f} GB/s, "
                "host {:8.3f} us".format(
                    "pinned" if pinned else "pageable",
                    nbytes / MIB,
                    median_ms,
                    gbps,
                    statistics.median(host_us),
                )
            )
            del source, target, host, destination
    pinned_rows = [
        row
        for row in results
        if row["pinned"] and row["bytes"] >= 64 * 1024
    ]
    xs = [row["bytes"] for row in pinned_rows]
    ys = [row["gpu_elapsed_ms"]["median"] / 1000.0 for row in pinned_rows]
    fitted = linear_fit(xs, ys)
    if fitted is None:
        fit = {
            "available": False,
            "reason": "need at least two distinct pinned sizes >= 64 KiB",
            "fit_min_bytes": 64 * 1024,
        }
    else:
        intercept_s, seconds_per_byte, r_squared = fitted
        fit = {
            "available": seconds_per_byte > 0,
            "regression_intercept_us": intercept_s * 1e6,
            "asymptotic_GBps": (
                1.0 / seconds_per_byte / 1e9
                if seconds_per_byte > 0
                else None
            ),
            "r_squared": r_squared,
            "fit_min_bytes": 64 * 1024,
            "note": (
                "The regression intercept is not a physical H2D launch "
                "overhead; use a batched copy-count benchmark for that."
            ),
        }
    return {
        "rows": results,
        "pinned_linear_fit": fit,
        "note": (
            "This is the framework-level PyTorch path. Small transfers include "
            "eager-dispatch gaps; use h2d_driver_bench.c for raw DMA and fixed "
            "Driver API enqueue cost. Pageable results are end-to-end staging "
            "measurements and are not evidence of asynchronous overlap."
        ),
    }


def cpu_memcpy_sweep(device, sizes, warmup, repetitions):
    del device
    if not sizes:
        return {
            "method": "single_thread_libc_memcpy_pageable_to_pinned_rotating_ring",
            "rows": [],
        }
    libc = ctypes.CDLL(None)
    memcpy = libc.memcpy
    memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    memcpy.restype = ctypes.c_void_p
    largest = max(sizes)
    ring_bytes = max(CPU_STAGING_RING_MIN_BYTES, 2 * largest)
    source = torch.empty(ring_bytes, dtype=torch.uint8)
    destination = torch.empty(
        ring_bytes, dtype=torch.uint8, pin_memory=True
    )
    source.fill_(0x5A)
    destination.zero_()
    rows = []
    for nbytes in sizes:
        slots = max(2, ring_bytes // nbytes)

        def copy_at(sequence):
            offset = (sequence % slots) * nbytes
            src_ptr = ctypes.c_void_p(source.data_ptr() + offset)
            dst_ptr = ctypes.c_void_p(destination.data_ptr() + offset)
            memcpy(dst_ptr, src_ptr, nbytes)

        for index in range(warmup):
            copy_at(index)
        times_ms = []
        for index in range(repetitions):
            started = time.perf_counter_ns()
            copy_at(warmup + index)
            ended = time.perf_counter_ns()
            times_ms.append((ended - started) / 1e6)
        median_ms = statistics.median(times_ms)
        gbps = nbytes / (median_ms / 1000.0) / 1e9
        row = {
            "bytes": nbytes,
            "elapsed_ms": summary(times_ms),
            "effective_GBps": gbps,
            "ring_bytes": ring_bytes,
            "ring_slots_for_size": slots,
        }
        rows.append(row)
        print(
            "CPU pageable->pinned {:9.3f} MiB: {:9.4f} ms, {:7.3f} GB/s".format(
                nbytes / MIB, median_ms, gbps
            )
        )
    del source, destination
    return {
        "method": "single_thread_libc_memcpy_pageable_to_pinned_rotating_ring",
        "ring_bytes": ring_bytes,
        "minimum_ring_bytes": CPU_STAGING_RING_MIN_BYTES,
        "note": (
            "Offsets rotate through a working set larger than the 128 MiB LLC; "
            "this measures one calling thread and is not aggregate DRAM bandwidth."
        ),
        "rows": rows,
    }


class LlamaLinearLayer:
    def __init__(self, device, slab=None):
        self.device = device
        if slab is None:
            self.slab = torch.empty(
                LAYER_WEIGHT_ELEMENTS, device=device, dtype=DTYPE
            )
            # Avoid NaN/Inf propagation from uninitialised BF16 bit patterns.
            self.slab.zero_()
        else:
            if (
                slab.device != torch.device("cuda", device)
                or slab.dtype != DTYPE
                or slab.numel() != LAYER_WEIGHT_ELEMENTS
            ):
                raise ValueError("provided projection slab has wrong device/dtype/size")
            self.slab = slab
        self.weights = {}
        offset = 0
        for name, shape in LLAMA_SHAPES:
            elements = math.prod(shape)
            self.weights[name] = self.slab[offset : offset + elements].view(shape)
            offset += elements
        assert offset == LAYER_WEIGHT_ELEMENTS

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


def compute_sweep(layer, rows_per_call, warmup, repetitions):
    results = []
    for rows in rows_per_call:
        x = torch.zeros((rows, 2048), device=layer.device, dtype=DTYPE)

        def compute():
            return layer(x)

        with torch.inference_mode():
            elapsed_ms, host_us = event_measure(compute, warmup, repetitions)
        median_ms = statistics.median(elapsed_ms)
        flops = LAYER_LINEAR_FLOPS_PER_ROW * rows
        tflops = flops / (median_ms / 1000.0) / 1e12
        weight_read_gbps = (
            LAYER_WEIGHT_ELEMENTS * DTYPE_BYTES
            / (median_ms / 1000.0)
            / 1e9
        )
        result = {
            "rows_M": rows,
            "projection_linear_flops": flops,
            "gpu_elapsed_ms": summary(elapsed_ms),
            "host_enqueue_us": summary(host_us),
            "effective_TFLOPs": tflops,
            "nominal_weight_bytes_per_elapsed_GBps": weight_read_gbps,
        }
        results.append(result)
        print(
            "Compute M={:5d}: {:9.4f} ms, {:8.3f} TFLOP/s, "
            "nominal-weight/time {:8.3f} GB/s".format(
                rows, median_ms, tflops, weight_read_gbps
            )
        )
        del x
    return results


PAIR_MODES = (
    "isolated_copy",
    "isolated_compute",
    "sequential",
    "overlap",
)


def make_pair_events(coordinator):
    events = {
        name: torch.cuda.Event(enable_timing=True)
        for name in (
            "start",
            "copy_begin",
            "copy_end",
            "compute_begin",
            "compute_end",
            "finish",
        )
    }
    # CUDA Events are created lazily on first record.  Prime every event before
    # a measured interval so event creation cannot become a host-side hole in it.
    for event in events.values():
        event.record(coordinator)
    events["finish"].synchronize()
    return events


def one_pair_sample(
    layer,
    x,
    host_slab,
    destination,
    mode,
    copy_stream,
    compute_stream,
    coordinator,
    events,
):
    if mode not in PAIR_MODES:
        raise ValueError("unknown pair mode {}".format(mode))
    has_copy = mode != "isolated_compute"
    has_compute = mode != "isolated_copy"

    events["start"].record(coordinator)
    if has_copy:
        copy_stream.wait_event(events["start"])
        with torch.cuda.stream(copy_stream):
            events["copy_begin"].record(copy_stream)
            destination.copy_(host_slab, non_blocking=True)
            events["copy_end"].record(copy_stream)

    if has_compute:
        if mode == "sequential":
            compute_stream.wait_event(events["copy_end"])
        else:
            compute_stream.wait_event(events["start"])
        with torch.cuda.stream(compute_stream):
            events["compute_begin"].record(compute_stream)
            layer(x)
            events["compute_end"].record(compute_stream)

    if has_copy:
        coordinator.wait_event(events["copy_end"])
    if has_compute:
        coordinator.wait_event(events["compute_end"])
    events["finish"].record(coordinator)
    events["finish"].synchronize()

    sample = {
        "span_ms": events["start"].elapsed_time(events["finish"]),
    }
    if has_copy:
        sample["copy_duration_ms"] = events["copy_begin"].elapsed_time(
            events["copy_end"]
        )
        sample["copy_start_offset_ms"] = events["start"].elapsed_time(
            events["copy_begin"]
        )
    if has_compute:
        sample["compute_duration_ms"] = events["compute_begin"].elapsed_time(
            events["compute_end"]
        )
        sample["compute_start_offset_ms"] = events["start"].elapsed_time(
            events["compute_begin"]
        )
    return sample


def summarize_samples(samples):
    return {
        key: summary([sample[key] for sample in samples])
        for key in samples[0]
    }


def overlap_sweep(
    layer,
    rows_per_call,
    warmup,
    repetitions,
    copy_stream,
    compute_stream,
):
    host_slab = torch.empty(
        LAYER_TOTAL_WEIGHT_ELEMENTS, dtype=DTYPE, pin_memory=True
    )
    host_slab.zero_()
    destination = torch.empty(
        LAYER_TOTAL_WEIGHT_ELEMENTS,
        dtype=DTYPE,
        device=layer.device,
    )
    coordinator = torch.cuda.current_stream(layer.device)
    torch.cuda.synchronize(layer.device)
    events = make_pair_events(coordinator)
    results = []
    for rows in rows_per_call:
        x = torch.zeros((rows, 2048), device=layer.device, dtype=DTYPE)
        torch.cuda.synchronize(layer.device)
        samples_by_mode = {mode: [] for mode in PAIR_MODES}
        with torch.inference_mode():
            # Prime shape-specific allocator and cuBLAS state on the same
            # persistent streams used by every measured mode.
            for mode in PAIR_MODES:
                one_pair_sample(
                    layer,
                    x,
                    host_slab,
                    destination,
                    mode,
                    copy_stream,
                    compute_stream,
                    coordinator,
                    events,
                )
            for _ in range(warmup):
                for mode in PAIR_MODES:
                    one_pair_sample(
                        layer,
                        x,
                        host_slab,
                        destination,
                        mode,
                        copy_stream,
                        compute_stream,
                        coordinator,
                        events,
                    )
            # Rotate order each round to avoid assigning a fixed thermal/DVFS
            # position to either baseline or overlap.
            for repetition in range(repetitions):
                shift = repetition % len(PAIR_MODES)
                order = PAIR_MODES[shift:] + PAIR_MODES[:shift]
                for mode in order:
                    samples_by_mode[mode].append(
                        one_pair_sample(
                            layer,
                            x,
                            host_slab,
                            destination,
                            mode,
                            copy_stream,
                            compute_stream,
                            coordinator,
                            events,
                        )
                    )

        modes = {
            mode: summarize_samples(samples_by_mode[mode])
            for mode in PAIR_MODES
        }
        copy_only = modes["isolated_copy"]["copy_duration_ms"]["median"]
        compute_only = modes["isolated_compute"]["compute_duration_ms"]["median"]
        copy_only_span = modes["isolated_copy"]["span_ms"]["median"]
        compute_only_span = modes["isolated_compute"]["span_ms"]["median"]
        sequential = modes["sequential"]["span_ms"]["median"]
        concurrent = modes["overlap"]["span_ms"]["median"]
        concurrent_copy = modes["overlap"]["copy_duration_ms"]["median"]
        concurrent_compute = modes["overlap"]["compute_duration_ms"]["median"]
        sequential_copy = modes["sequential"]["copy_duration_ms"]["median"]
        sequential_compute = modes["sequential"]["compute_duration_ms"]["median"]
        hidden_fraction_raw = (
            (copy_only_span + compute_only_span - concurrent)
            / min(copy_only_span, compute_only_span)
        )
        metrics = {
            "speedup_vs_sequential": sequential / concurrent,
            "saved_ms_vs_sequential": sequential - concurrent,
            "copy_slowdown_during_overlap": concurrent_copy / copy_only,
            "compute_slowdown_during_overlap": concurrent_compute / compute_only,
            "copy_slowdown_vs_sequential_component": (
                concurrent_copy / sequential_copy
            ),
            "compute_slowdown_vs_sequential_component": (
                concurrent_compute / sequential_compute
            ),
            "hidden_fraction": min(1.0, hidden_fraction_raw),
            "hidden_fraction_raw_estimate": hidden_fraction_raw,
            "hidden_fraction_timing_scope": (
                "common-start-to-finish spans for isolated copy, isolated "
                "compute, and overlap; hidden_fraction is upper-capped at 1 "
                "because timing noise can make the raw estimate exceed 1"
            ),
        }
        results.append(
            {
                "rows_M": rows,
                "copy_bytes_exact_layer_with_norms": LAYER_BYTES_WITH_NORMS,
                "compute_scope": "projection_linear_only",
                "modes": modes,
                "metrics": metrics,
                "memory": {
                    "allocated_bytes": torch.cuda.memory_allocated(layer.device),
                    "reserved_bytes": torch.cuda.memory_reserved(layer.device),
                },
            }
        )
        print(
            "Pair M={:5d}: copy-only {:8.4f} ms, compute-only {:8.4f} ms, "
            "sequential {:8.4f} ms, overlap {:8.4f} ms".format(
                rows, copy_only, compute_only, sequential, concurrent
            )
        )
        print(
            "  => speedup {:.3f}x, hidden {:.3f}, copy slowdown {:.3f}x, "
            "compute slowdown {:.3f}x".format(
                metrics["speedup_vs_sequential"],
                metrics["hidden_fraction"],
                metrics["copy_slowdown_during_overlap"],
                metrics["compute_slowdown_during_overlap"],
            )
        )
        del x
    return {
        "persistent_streams": True,
        "copy_source": "pinned_cpu_exact_layer_with_norms",
        "copy_destination_is_next_layer_slot_while_compute_uses_current_layer": True,
        "clock_caveat": (
            "GPU clocks are not locked. Tiny-M isolated compute can start in "
            "a different power state from a compute component preceded by "
            "H2D; compare both isolated and sequential-component slowdown."
        ),
        "rows": results,
    }


def make_pipeline_events(stage_count, coordinator):
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    ready = [
        torch.cuda.Event(enable_timing=True) for _ in range(stage_count)
    ]
    free = [
        torch.cuda.Event(enable_timing=True) for _ in range(stage_count)
    ]
    for event in [start, finish] + ready + free:
        event.record(coordinator)
    free[-1].synchronize()
    return start, finish, ready, free


def one_pipeline_sample(
    x,
    host_slab,
    device_slots,
    slot_layers,
    copy_stream,
    compute_stream,
    coordinator,
    events,
    overlap,
):
    start, finish, ready, free = events
    stage_count = len(ready)
    start.record(coordinator)
    copy_stream.wait_event(start)
    compute_stream.wait_event(start)
    activation = x
    host_started = time.perf_counter_ns()
    for stage in range(stage_count):
        slot = stage % 2
        if overlap and stage >= 2:
            copy_stream.wait_event(free[stage - 2])
        elif not overlap and stage >= 1:
            copy_stream.wait_event(free[stage - 1])
        with torch.cuda.stream(copy_stream):
            device_slots[slot].copy_(host_slab, non_blocking=True)
            ready[stage].record(copy_stream)
        compute_stream.wait_event(ready[stage])
        with torch.cuda.stream(compute_stream):
            activation = slot_layers[slot](activation)[0]
            free[stage].record(compute_stream)
    coordinator.wait_event(free[-1])
    finish.record(coordinator)
    host_ended = time.perf_counter_ns()
    finish.synchronize()
    total_ms = start.elapsed_time(finish)
    completion_intervals = [
        free[index - 1].elapsed_time(free[index])
        for index in range(1, stage_count)
    ]
    return {
        "mode": "overlap" if overlap else "sequential",
        "total_ms": total_ms,
        "average_ms_per_stage_including_fill_drain": total_ms / stage_count,
        "host_enqueue_us": (host_ended - host_started) / 1000.0,
        "stage_completion_intervals_ms": completion_intervals,
    }


def pipeline_sweep(
    device,
    rows_per_call,
    stage_count,
    warmup,
    repetitions,
    copy_stream,
    compute_stream,
):
    host_slab = torch.empty(
        LAYER_TOTAL_WEIGHT_ELEMENTS, dtype=DTYPE, pin_memory=True
    )
    host_slab.zero_()
    device_slots = [
        torch.empty(
            LAYER_TOTAL_WEIGHT_ELEMENTS, dtype=DTYPE, device=device
        )
        for _ in range(2)
    ]
    slot_layers = [
        LlamaLinearLayer(device, slab=slot[:LAYER_WEIGHT_ELEMENTS])
        for slot in device_slots
    ]
    coordinator = torch.cuda.current_stream(device)
    torch.cuda.synchronize(device)
    events = make_pipeline_events(stage_count, coordinator)
    rows_results = []
    for rows in rows_per_call:
        x = torch.zeros((rows, 2048), dtype=DTYPE, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples_by_mode = {"sequential": [], "overlap": []}
        with torch.inference_mode():
            for overlap in (False, True):
                for _ in range(max(1, warmup)):
                    one_pipeline_sample(
                        x,
                        host_slab,
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
                    mode = "overlap" if overlap else "sequential"
                    samples_by_mode[mode].append(
                        one_pipeline_sample(
                            x,
                            host_slab,
                            device_slots,
                            slot_layers,
                            copy_stream,
                            compute_stream,
                            coordinator,
                            events,
                            overlap,
                        )
                    )

        modes = {}
        for mode, samples in samples_by_mode.items():
            totals = [sample["total_ms"] for sample in samples]
            average_stages = [
                sample["average_ms_per_stage_including_fill_drain"]
                for sample in samples
            ]
            host_enqueue = [sample["host_enqueue_us"] for sample in samples]
            intervals = [
                interval
                for sample in samples
                for interval in sample["stage_completion_intervals_ms"]
            ]
            modes[mode] = {
                "total_ms": summary(totals),
                "average_ms_per_stage_including_fill_drain": summary(
                    average_stages
                ),
                "stage_completion_interval_ms": summary(intervals),
                "host_enqueue_us": summary(host_enqueue),
            }
        sequential_ms = modes["sequential"]["total_ms"]["median"]
        overlap_ms = modes["overlap"]["total_ms"]["median"]
        row = {
            "rows_M": rows,
            "modes": modes,
            "speedup_vs_sequential": sequential_ms / overlap_ms,
            "saved_ms_vs_sequential": sequential_ms - overlap_ms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        rows_results.append(row)
        print(
            "Pipeline M={:5d}: {} stages sequential {:9.4f} ms, "
            "overlap {:9.4f} ms, speedup {:.3f}x".format(
                rows,
                stage_count,
                sequential_ms,
                overlap_ms,
                row["speedup_vs_sequential"],
            )
        )
        del x
    return {
        "stage_count": stage_count,
        "device_weight_slots": 2,
        "slot_bytes": LAYER_BYTES_WITH_NORMS,
        "source_reused_for_all_synthetic_stages": True,
        "sync_policy": "one finish-event synchronize per full pipeline sample",
        "overlap_dependency": (
            "copy(i>=2) waits compute_free(i-2); compute(i) waits ready(i)"
        ),
        "sequential_dependency": (
            "copy(i>=1) waits compute_free(i-1); compute(i) waits ready(i)"
        ),
        "compute_scope": "projection_linear_only",
        "rows": rows_results,
    }


def parse_sizes(text):
    suffixes = {"K": 1024, "M": MIB, "G": GIB}
    sizes = []
    for item in text.split(","):
        item = item.strip().upper()
        if not item:
            continue
        multiplier = 1
        if item[-1:] in suffixes:
            multiplier = suffixes[item[-1]]
            item = item[:-1]
        sizes.append(int(float(item) * multiplier))
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--h2d-sizes",
        default=(
            "0,1,4K,64K,1M,2M,4M,8M,16M,20M,32M,64M,96M,"
            "116M,116.0078125M,256M"
        ),
    )
    parser.add_argument(
        "--compute-rows",
        default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096",
    )
    parser.add_argument(
        "--overlap-rows", default="1,8,32,128,512,1024,2048,4096"
    )
    parser.add_argument("--pipeline-stages", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/torch_llama32_1b_bench.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this PyTorch environment")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise SystemExit("CUDA device {} does not exist".format(args.device))
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.pipeline_stages < 2:
        raise SystemExit("--pipeline-stages must be at least 2")
    torch.cuda.set_device(args.device)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("selected CUDA device does not support BF16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    h2d_sizes = parse_sizes(args.h2d_sizes)
    compute_rows = [int(item) for item in args.compute_rows.split(",")]
    overlap_rows = [int(item) for item in args.overlap_rows.split(",")]
    if any(size < 0 for size in h2d_sizes):
        raise SystemExit("H2D sizes must be non-negative")
    if any(rows <= 0 for rows in compute_rows + overlap_rows):
        raise SystemExit("compute/overlap rows must be positive")

    metadata = environment(args.device)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(
        "Llama layer: {} params, {:.6f} MiB BF16 including norms".format(
            LAYER_WEIGHT_ELEMENTS + NORM_ELEMENTS,
            LAYER_BYTES_WITH_NORMS / MIB,
        )
    )

    results = {
        "environment": metadata,
        "model": {
            "id": "meta-llama/Llama-3.2-1B",
            "synthetic_weights": True,
            "dtype": str(DTYPE),
            "layer_linear_weight_elements": LAYER_WEIGHT_ELEMENTS,
            "layer_total_weight_elements": LAYER_WEIGHT_ELEMENTS + NORM_ELEMENTS,
            "layer_bytes_with_norms": LAYER_BYTES_WITH_NORMS,
            "layer_linear_flops_per_row": LAYER_LINEAR_FLOPS_PER_ROW,
            "shapes": [
                {"name": name, "shape": list(shape)}
                for name, shape in LLAMA_SHAPES
            ],
        },
        "settings": vars(args).copy(),
    }
    results["settings"]["output"] = str(args.output)

    print("\n== H2D sweep ==")
    results["h2d"] = h2d_sweep(
        args.device, h2d_sizes, args.warmup, args.repetitions
    )

    print("\n== CPU pageable -> pinned staging memcpy ==")
    cpu_sizes = [
        size for size in h2d_sizes if 1 * MIB <= size <= 256 * MIB
    ]
    results["cpu_staging_memcpy"] = cpu_memcpy_sweep(
        args.device, cpu_sizes, args.warmup, args.repetitions
    )

    print("\n== Exact-shape Llama linear stack ==")
    layer = LlamaLinearLayer(args.device)
    results["compute"] = compute_sweep(
        layer, compute_rows, args.warmup, args.repetitions
    )

    print("\n== One-layer H2D versus current-layer compute ==")
    copy_stream = torch.cuda.Stream(device=args.device)
    compute_stream = torch.cuda.Stream(device=args.device)
    overlap_repetitions = max(5, args.repetitions // 2)
    results["settings"]["overlap_repetitions"] = overlap_repetitions
    results["overlap"] = overlap_sweep(
        layer,
        overlap_rows,
        args.warmup,
        overlap_repetitions,
        copy_stream,
        compute_stream,
    )

    print("\n== Multi-stage two-slot pipeline ==")
    pipeline_repetitions = max(3, args.repetitions // 4)
    results["settings"]["pipeline_repetitions"] = pipeline_repetitions
    results["pipeline"] = pipeline_sweep(
        args.device,
        overlap_rows,
        args.pipeline_stages,
        args.warmup,
        pipeline_repetitions,
        copy_stream,
        compute_stream,
    )

    torch.cuda.synchronize(args.device)
    results["environment"]["nvidia_smi_after_benchmark"] = (
        nvidia_smi_snapshot()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    print("\nWrote {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
