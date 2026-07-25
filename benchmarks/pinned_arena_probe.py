#!/usr/bin/env python3
"""Check whether a model-sized CUDA page-locked CPU weight arena is viable."""

import argparse
import ctypes
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

# Exact tensor payload of meta-llama/Llama-3.2-1B model.safetensors.
LLAMA32_1B_TENSOR_BYTES = 2_471_628_800
CUDA_SUCCESS = 0


def cuda_check(cuda, result, operation):
    if result == CUDA_SUCCESS:
        return
    name_pointer = ctypes.c_char_p()
    message_pointer = ctypes.c_char_p()
    cuda.cuGetErrorName(result, ctypes.byref(name_pointer))
    cuda.cuGetErrorString(result, ctypes.byref(message_pointer))
    name = name_pointer.value.decode() if name_pointer.value else "unknown"
    message = (
        message_pointer.value.decode() if message_pointer.value else "unknown"
    )
    raise RuntimeError(
        "{} failed: {} ({}) — {}".format(operation, result, name, message)
    )


def load_cuda():
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cuDriverGetVersion.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_int,
    ]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuMemHostAlloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    cuda.cuMemHostAlloc.restype = ctypes.c_int
    cuda.cuMemFreeHost.argtypes = [ctypes.c_void_p]
    cuda.cuMemFreeHost.restype = ctypes.c_int
    cuda.cuGetErrorName.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    cuda.cuGetErrorName.restype = ctypes.c_int
    cuda.cuGetErrorString.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    cuda.cuGetErrorString.restype = ctypes.c_int
    return cuda


def meminfo():
    values = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as source:
        for line in source:
            key, value = line.split(":", 1)
            fields = value.split()
            if fields:
                values[key] = int(fields[0]) * 1024
    return {
        key: values.get(key)
        for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=LLAMA32_1B_TENSOR_BYTES)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/pinned_arena_probe.json")
    )
    args = parser.parse_args()
    if args.bytes <= 0:
        raise SystemExit("--bytes must be positive")

    cuda = load_cuda()
    cuda_check(cuda, cuda.cuInit(0), "cuInit")
    driver_version = ctypes.c_int()
    cuda_check(
        cuda,
        cuda.cuDriverGetVersion(ctypes.byref(driver_version)),
        "cuDriverGetVersion",
    )
    device = ctypes.c_int()
    cuda_check(
        cuda,
        cuda.cuDeviceGet(ctypes.byref(device), args.device),
        "cuDeviceGet",
    )
    context = ctypes.c_void_p()
    cuda_check(
        cuda,
        cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device.value),
        "cuCtxCreate_v2",
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "cuda_driver_api_version": driver_version.value,
        "device_index": args.device,
        "requested_bytes": args.bytes,
        "requested_gib": args.bytes / (1024 ** 3),
        "model_id": "meta-llama/Llama-3.2-1B",
        "memory_before": meminfo(),
    }

    arena = ctypes.c_void_p()
    allocation_start = time.perf_counter_ns()
    cuda_check(
        cuda,
        cuda.cuMemHostAlloc(ctypes.byref(arena), args.bytes, 0),
        "cuMemHostAlloc",
    )
    allocation_end = time.perf_counter_ns()
    result["allocation_seconds"] = (
        allocation_end - allocation_start
    ) / 1e9
    result["allocation_api"] = "cuMemHostAlloc"

    libc = ctypes.CDLL(None)
    memset = libc.memset
    memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    memset.restype = ctypes.c_void_p
    touch_start = time.perf_counter_ns()
    memset(arena, 0, args.bytes)
    touch_end = time.perf_counter_ns()
    result["first_touch_seconds"] = (touch_end - touch_start) / 1e9
    result["first_touch_GBps"] = (
        args.bytes / result["first_touch_seconds"] / 1e9
    )
    result["memory_while_allocated"] = meminfo()

    free_start = time.perf_counter_ns()
    cuda_check(cuda, cuda.cuMemFreeHost(arena), "cuMemFreeHost")
    free_end = time.perf_counter_ns()
    result["free_seconds"] = (free_end - free_start) / 1e9
    cuda_check(cuda, cuda.cuCtxDestroy_v2(context), "cuCtxDestroy_v2")
    gc.collect()
    result["memory_after_free"] = meminfo()
    result["status"] = "ok"
    result["note"] = (
        "Allocation/free use the CUDA Driver API directly; first touch uses "
        "one libc memset call. This probes feasibility, not aggregate CPU "
        "DRAM bandwidth."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
