#!/usr/bin/env python3
"""Long-running KV lifecycle validation and qualification-ready reporting.

The default ``logic`` profile is intentionally small and CPU-only.  It proves
that the harness and ownership counters close; it is not a CUDA or real-model
qualification.  ``cuda-smoke`` remains a smoke profile even on an idle GPU.
The ``full`` profile expands the synthetic matrix and records unavailable
real-model cases as ``SKIPPED_WITH_REASON`` instead of treating them as PASS.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from layer_streaming.kv import PagedKVRuntime
from layer_streaming.kv.resource_audit import (
    BLOCKED,
    KVResourceSnapshot,
    compare_resource_snapshots,
)
from layer_streaming.kv_policy import (
    KVAccuracy,
    KVDataType,
    KVPolicy,
    KVReusePolicy,
    KVSelectionPolicy,
    KVStoragePolicy,
)


SCHEMA_VERSION = 2
PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED_WITH_REASON"
NOT_QUALIFIED = "NOT_QUALIFIED"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except BaseException:
        return None


def _nvidia_environment():
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except BaseException as error:
        return {
            "available": False,
            "processes": [],
            "reason": "{}: {}".format(type(error).__name__, error),
        }
    processes = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",", 3)]
        if len(fields) != 4:
            continue
        try:
            pid = int(fields[1])
            used_memory = int(fields[2])
        except ValueError:
            continue
        processes.append(
            {
                "gpu_uuid": fields[0],
                "pid": pid,
                "used_memory_mib": used_memory,
                "process_name": fields[3],
                "is_current_process": pid == os.getpid(),
            }
        )
    external = [item for item in processes if not item["is_current_process"]]
    return {
        "available": True,
        "processes": processes,
        "external_compute_processes": len(external),
        "exclusive_at_capture": not external,
    }


def capture_environment():
    result = {
        "captured_at": _utc_now(),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "nvidia_smi": _nvidia_environment(),
    }
    if torch.cuda.is_available():
        result["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(
                    torch.cuda.get_device_properties(index).total_memory
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
    return result


def _process_rss_bytes():
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except BaseException:
        # ru_maxrss is a peak, not current RSS, so using it as a substitute
        # would make drift results misleading.
        return None


def _prefix_stats(runtime):
    cache = getattr(runtime, "prefix_cache", None)
    if cache is None or not hasattr(cache, "stats"):
        return {
            "entries": 0,
            "pages": 0,
            "bytes": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "replacements": 0,
        }
    stats = cache.stats()
    return {
        "entries": int(stats.get("prefix_entries", 0)),
        "pages": int(stats.get("prefix_pages", 0)),
        "bytes": int(stats.get("prefix_bytes", 0)),
        "hits": int(stats.get("prefix_hits", 0)),
        "misses": int(stats.get("prefix_misses", 0)),
        "evictions": int(stats.get("prefix_evictions", 0)),
        "replacements": int(stats.get("prefix_replacements", 0)),
    }


def _rgkv_stats(runtime):
    selection = getattr(runtime, "selection", None)
    index = getattr(selection, "index", None)
    if index is None or not hasattr(index, "stats"):
        return {"records": 0, "references": 0, "bytes": 0}
    stats = index.stats()
    return {
        "records": int(stats.get("records", 0)),
        "references": int(stats.get("references", 0)),
        "bytes": int(stats.get("compact_bytes", 0)),
    }


def capture_snapshot(runtime, sample_index, token_index, *, tier_store=None):
    if tier_store is None:
        tier_store = getattr(runtime, "active_tier", None)
    audit = KVResourceSnapshot.capture(
        runtime,
        sample_index,
        token_index,
        tier_store=tier_store,
    )
    pool = runtime.page_pool.profile()
    profile = runtime.profile_stats()
    requests = tuple(runtime.request_table.values())
    tier = tier_store.stats() if tier_store is not None else {}
    if runtime.device.type == "cuda":
        cuda_allocated = int(torch.cuda.memory_allocated(runtime.device))
        cuda_reserved = int(torch.cuda.memory_reserved(runtime.device))
    else:
        cuda_allocated = 0
        cuda_reserved = 0
    return {
        "sample_index": int(sample_index),
        "token_index": int(token_index),
        "monotonic_seconds": time.monotonic(),
        "kv": {
            "page_total": int(pool["total_pages"]),
            "page_free": int(pool["free_pages"]),
            "page_allocated": int(pool["allocated_pages"]),
            "page_peak_allocated": int(pool["peak_allocated_pages"]),
            "ref_count": int(pool["total_ref_count"]),
            "logical_owner_count": int(pool["logical_owner_count"]),
            "pin_count": int(pool["total_pin_count"]),
            "inflight_compute": int(pool["inflight_compute"]),
            "inflight_io": int(pool["inflight_io"]),
            "pending_transactions": sum(
                state.pending_append is not None for state in requests
            ),
            "pending_append_fences": int(
                profile.get("pending_append_fences", 0)
            ),
            "pending_attention_fences": int(
                profile.get("pending_attention_fences", 0)
            ),
            "active_requests": int(profile.get("active_requests", 0)),
        },
        "prefix": _prefix_stats(runtime),
        "rgkv": _rgkv_stats(runtime),
        "tier": {
            "gpu_used_pages": int(tier.get("gpu_kv_used_pages", 0)),
            "cpu_used_bytes": int(tier.get("cpu_kv_used_bytes", 0)),
            "pending_operations": int(tier.get("pending_tier_operations", 0)),
            "active_prefetch_groups": int(tier.get("active_prefetch_groups", 0)),
        },
        "system": {
            "cuda_allocated_bytes": cuda_allocated,
            "cuda_reserved_bytes": cuda_reserved,
            "cpu_rss_bytes": _process_rss_bytes(),
            "pinned_rss_bytes": int(tier.get("cpu_kv_used_bytes", 0)),
            "pinned_rss_source": (
                "active_tier_store" if tier_store is not None else "no_active_tier"
            ),
            "event_count": int(profile.get("cuda_event_count", 0)),
            "thread_count": int(threading.active_count()),
            "future_count": int(tier.get("pending_tier_operations", 0)),
        },
        "resource_audit": audit.as_dict(),
    }


def _walk_numeric(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            path = "{}.{}".format(prefix, key) if prefix else str(key)
            yield from _walk_numeric(item, path)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def summarize_snapshots(snapshots):
    if len(snapshots) < 2:
        raise ValueError("at least two snapshots are required")
    baseline = snapshots[0]
    final = snapshots[-1]
    baseline_flat = dict(_walk_numeric(baseline))
    final_flat = dict(_walk_numeric(final))
    peak = {}
    for snapshot in snapshots:
        for key, value in _walk_numeric(snapshot):
            if key.startswith(("sample_index", "token_index", "monotonic_seconds")):
                continue
            if value is not None:
                peak[key] = max(value, peak.get(key, value))
    delta = {
        key: final_flat[key] - value
        for key, value in baseline_flat.items()
        if key in final_flat
        and not key.startswith(("sample_index", "token_index", "monotonic_seconds"))
    }
    return baseline, peak, final, delta


_ZERO_FINAL_FIELDS = (
    "kv.page_allocated",
    "kv.ref_count",
    "kv.logical_owner_count",
    "kv.pin_count",
    "kv.inflight_compute",
    "kv.inflight_io",
    "kv.pending_transactions",
    "kv.pending_append_fences",
    "kv.pending_attention_fences",
    "kv.active_requests",
    "tier.gpu_used_pages",
    "tier.cpu_used_bytes",
    "tier.pending_operations",
    "tier.active_prefetch_groups",
    "system.future_count",
)


def assess_drift(baseline, final):
    if "resource_audit" in baseline and "resource_audit" in final:
        comparison = compare_resource_snapshots(
            KVResourceSnapshot.from_dict(baseline["resource_audit"]),
            KVResourceSnapshot.from_dict(final["resource_audit"]),
        )
        failures = [
            "{} final drift is {}, expected closure".format(item.field, item.delta)
            for item in comparison.leaks
        ]
        if final["kv"]["page_free"] != final["kv"]["page_total"]:
            failures.append("final PagePool is not fully free")
        return failures
    baseline_flat = dict(_walk_numeric(baseline))
    final_flat = dict(_walk_numeric(final))
    failures = []
    for field in _ZERO_FINAL_FIELDS:
        value = final_flat.get(field, 0)
        if value != 0:
            failures.append("{} final value is {}, expected 0".format(field, value))
    if final_flat.get("kv.page_free") != final_flat.get("kv.page_total"):
        failures.append("final PagePool is not fully free")
    if final_flat.get("rgkv.records", 0) != baseline_flat.get("rgkv.records", 0):
        failures.append("RGKV record count did not return to baseline")
    if final_flat.get("rgkv.references", 0) != baseline_flat.get("rgkv.references", 0):
        failures.append("RGKV reference count did not return to baseline")
    if final_flat.get("prefix.entries", 0) != baseline_flat.get("prefix.entries", 0):
        failures.append("Prefix entry count did not return to baseline")
    if final_flat.get("prefix.pages", 0) != baseline_flat.get("prefix.pages", 0):
        failures.append("Prefix page count did not return to baseline")
    if final_flat.get("system.thread_count", 0) > baseline_flat.get("system.thread_count", 0):
        failures.append("thread count increased")
    return failures


def _make_runtime(
    device,
    max_length,
    *,
    reuse=KVReusePolicy.REQUEST_ONLY,
    selection=KVSelectionPolicy.DENSE,
    max_prefix_pages=None,
    active_tier=False,
):
    device = torch.device(device)
    page_size = 16
    page_count = max(4, int(math.ceil(int(max_length) / page_size)) + 2)
    dtype = (
        torch.float16
        if active_tier and device.type == "cpu"
        else torch.float32
        if device.type == "cpu"
        else torch.bfloat16
    )
    selection = KVSelectionPolicy(selection)
    logical_page_bytes = (
        2 * 1 * 1 * page_size * 4 * torch.empty((), dtype=dtype).element_size()
    )
    hot_pages = min(3, page_count)
    policy = KVPolicy(
        accuracy=(
            KVAccuracy.EXACT
            if selection == KVSelectionPolicy.DENSE
            else KVAccuracy.SPARSE
        ),
        dtype=(
            KVDataType.FP16 if dtype == torch.float16 else KVDataType.BF16
        ),
        reuse=reuse,
        selection=selection,
        attention_backend="reference_paged_exact",
        page_size=page_size,
        page_budget=2 if selection != KVSelectionPolicy.DENSE else 0,
        recent_window=page_size if selection != KVSelectionPolicy.DENSE else 0,
        storage=(KVStoragePolicy.GPU_CPU if active_tier else KVStoragePolicy.GPU),
        rgkv_scorer=(
            "torch_tensorized"
            if selection == KVSelectionPolicy.RGKV
            else "cpu_reference"
        ),
        gpu_hot_budget_bytes=(hot_pages * logical_page_bytes if active_tier else 0),
        cpu_budget_bytes=(page_count * logical_page_bytes if active_tier else 0),
        gpu_high_watermark_bytes=(
            hot_pages * logical_page_bytes if active_tier else 0
        ),
        gpu_low_watermark_bytes=(
            max(0, hot_pages - 1) * logical_page_bytes if active_tier else 0
        ),
    )
    return PagedKVRuntime(
        layer_count=1,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
        page_count=page_count,
        page_size=page_size,
        dtype=dtype,
        device=device,
        policy=policy,
        allow_reference=True,
        max_prefix_pages=max_prefix_pages,
        tier_tensor_factory=(
            (lambda shape, **kwargs: torch.empty(
                shape, dtype=kwargs["dtype"], device="cpu"
            ))
            if active_tier and device.type == "cpu"
            else None
        ),
    )


def _append(runtime, state, token_count, value=1.0):
    tensor = torch.full(
        (int(token_count), runtime.num_kv_heads, runtime.head_dim),
        float(value),
        dtype=runtime.dtype,
        device=runtime.device,
    )
    for layer in range(runtime.layer_count):
        runtime.append((state,), layer, tensor, tensor, (int(token_count),))


def _execute_case(case_id, name, category, config, operation):
    started = time.perf_counter()
    try:
        snapshots, metrics = operation()
        baseline, peak, final, delta = summarize_snapshots(snapshots)
        failures = assess_drift(baseline, final)
        status = PASS if not failures else FAIL
        return {
            "case_id": case_id,
            "name": name,
            "category": category,
            "status": status,
            "qualification": NOT_QUALIFIED,
            "qualification_reason": (
                "logic/smoke evidence only; exclusive CUDA and real-model gates remain"
            ),
            "reason": None if not failures else "; ".join(failures),
            "duration_seconds": time.perf_counter() - started,
            "config": config,
            "baseline": baseline,
            "peak": peak,
            "final": final,
            "delta": delta,
            "metrics": metrics,
            "samples": snapshots,
        }
    except BaseException as error:
        return {
            "case_id": case_id,
            "name": name,
            "category": category,
            "status": FAIL,
            "qualification": NOT_QUALIFIED,
            "qualification_reason": "case failed before qualification review",
            "reason": "{}: {}".format(type(error).__name__, error),
            "duration_seconds": time.perf_counter() - started,
            "config": config,
            "baseline": None,
            "peak": None,
            "final": None,
            "delta": None,
            "metrics": {},
            "samples": [],
            "traceback": traceback.format_exc(),
        }


def skipped_case(case_id, name, category, reason, config=None):
    return {
        "case_id": case_id,
        "name": name,
        "category": category,
        "status": SKIPPED,
        "qualification": NOT_QUALIFIED,
        "qualification_reason": str(reason),
        "reason": str(reason),
        "duration_seconds": 0.0,
        "config": dict(config or {}),
        "baseline": None,
        "peak": None,
        "final": None,
        "delta": None,
        "metrics": {},
        "samples": [],
    }


def blocked_case(case_id, name, category, reason, config=None):
    result = skipped_case(case_id, name, category, reason, config)
    result["status"] = BLOCKED
    return result


def _warm_cuda_runtime(runtime):
    """Move one-time CUDA/PyTorch allocations outside the drift baseline."""

    if runtime.device.type != "cuda":
        return
    state = runtime.create_request(2)
    try:
        _append(runtime, state, 1)
        runtime.quiesce()
    finally:
        runtime.release(state)
    torch.cuda.synchronize(runtime.device)
    gc.collect()
    runtime.page_pool.validate_invariants()


def run_session_cycles(cycles, decode_tokens, device, sample_every=1):
    cycles = int(cycles)
    decode_tokens = int(decode_tokens)
    runtime = _make_runtime(device, max_length=16 + decode_tokens)
    _warm_cuda_runtime(runtime)
    snapshots = [capture_snapshot(runtime, 0, 0)]
    completed = 0
    try:
        for cycle in range(cycles):
            state = runtime.create_request(16 + decode_tokens)
            _append(runtime, state, 4, value=cycle + 1)
            for token in range(decode_tokens):
                _append(runtime, state, 1, value=token + 1)
            runtime.quiesce()
            runtime.reset(state)
            runtime.page_pool.validate_invariants()
            runtime.release(state)
            runtime.page_pool.validate_invariants()
            completed += 1
            if (cycle + 1) % int(sample_every) == 0 or cycle + 1 == cycles:
                snapshots.append(
                    capture_snapshot(runtime, len(snapshots), completed * decode_tokens)
                )
    finally:
        runtime.close()
    if snapshots[-1]["kv"]["active_requests"]:
        snapshots.append(capture_snapshot(runtime, len(snapshots), completed * decode_tokens))
    return snapshots, {"completed_cycles": completed}


def run_long_decode(tokens, device, sample_every=1):
    tokens = int(tokens)
    runtime = _make_runtime(device, max_length=tokens + 4)
    _warm_cuda_runtime(runtime)
    snapshots = [capture_snapshot(runtime, 0, 0)]
    completed = 0
    try:
        state = runtime.create_request(tokens + 4)
        _append(runtime, state, 1)
        for token in range(tokens):
            _append(runtime, state, 1, value=token + 2)
            completed += 1
            if completed % int(sample_every) == 0:
                snapshots.append(capture_snapshot(runtime, len(snapshots), completed))
        runtime.quiesce()
        runtime.release(state)
        runtime.page_pool.validate_invariants()
        snapshots.append(capture_snapshot(runtime, len(snapshots), completed))
    finally:
        runtime.close()
    return snapshots, {"completed_decode_tokens": completed}


def run_context(context_tokens, device):
    context_tokens = int(context_tokens)
    runtime = _make_runtime(device, max_length=context_tokens)
    _warm_cuda_runtime(runtime)
    snapshots = [capture_snapshot(runtime, 0, 0)]
    try:
        state = runtime.create_request(context_tokens)
        _append(runtime, state, context_tokens)
        runtime.quiesce()
        snapshots.append(capture_snapshot(runtime, 1, context_tokens))
        runtime.release(state)
        runtime.page_pool.validate_invariants()
        snapshots.append(capture_snapshot(runtime, 2, context_tokens))
    finally:
        runtime.close()
    return snapshots, {"completed_context_tokens": context_tokens}


def run_fork_cow_rollback_cycles(cycles, device="cpu", sample_every=100):
    cycles = int(cycles)
    runtime = _make_runtime(device, max_length=32)
    snapshots = [capture_snapshot(runtime, 0, 0)]
    completed = 0
    try:
        parent = runtime.create_request(32)
        _append(runtime, parent, 3)
        for cycle in range(cycles):
            branch = runtime.fork(parent, max_length=32)
            _append(runtime, branch, 2, value=cycle + 2)
            runtime.rollback(branch, parent.sequence_length)
            runtime.discard_branch(branch)
            runtime.page_pool.validate_invariants()
            completed += 1
            if completed % int(sample_every) == 0 or completed == cycles:
                snapshots.append(capture_snapshot(runtime, len(snapshots), completed))
        runtime.release(parent)
        snapshots.append(capture_snapshot(runtime, len(snapshots), completed))
    finally:
        runtime.close()
    return snapshots, {"completed_fork_cow_rollback_cycles": completed}


def run_prefix_cycles(cycles, device="cpu", sample_every=100):
    cycles = int(cycles)
    runtime = _make_runtime(
        device,
        max_length=32,
        reuse=KVReusePolicy.PREFIX_MEMORY,
        max_prefix_pages=2,
    )
    snapshots = [capture_snapshot(runtime, 0, 0)]
    completed = 0
    try:
        for cycle in range(cycles):
            state = runtime.create_request(32)
            _append(runtime, state, 16, value=cycle + 1)
            runtime.register_prefix(state, [cycle] * 16)
            runtime.release(state)
            runtime.page_pool.validate_invariants()
            completed += 1
            if completed % int(sample_every) == 0 or completed == cycles:
                snapshots.append(capture_snapshot(runtime, len(snapshots), completed))
        runtime.prefix_cache.evict_all()
        runtime.page_pool.validate_invariants()
        snapshots.append(capture_snapshot(runtime, len(snapshots), completed))
    finally:
        runtime.close()
    return snapshots, {"completed_prefix_insert_evict_cycles": completed}


def run_rgkv_index_cycles(cycles, device="cpu", sample_every=100):
    cycles = int(cycles)
    runtime = _make_runtime(
        device,
        max_length=32,
        selection=KVSelectionPolicy.RGKV,
    )
    snapshots = [capture_snapshot(runtime, 0, 0)]
    completed = 0
    try:
        for cycle in range(cycles):
            state = runtime.create_request(32)
            _append(runtime, state, 17, value=cycle + 1)
            runtime.release(state)
            runtime.page_pool.validate_invariants()
            completed += 1
            if completed % int(sample_every) == 0 or completed == cycles:
                snapshots.append(capture_snapshot(runtime, len(snapshots), completed))
        snapshots.append(capture_snapshot(runtime, len(snapshots), completed))
    finally:
        runtime.close()
    return snapshots, {"completed_rgkv_index_cycles": completed}


def run_rgkv_active_tier_session_cycles(
    cycles, device="cpu", sample_every=10
):
    """Exercise RGKV selection and real GPU/CPU Location orchestration.

    The CPU tensor factory makes this a deterministic lifecycle validation,
    not CUDA evidence.  Four logical pages deliberately exceed the three-page
    hot cache, and tied summaries select an old cold page plus the recent page,
    forcing selected-only eviction/prefetch before attention.
    """

    cycles = int(cycles)
    runtime = _make_runtime(
        device,
        max_length=96,
        selection=KVSelectionPolicy.RGKV,
        active_tier=True,
    )
    snapshots = [capture_snapshot(runtime, 0, 0)]
    completed = 0
    total_attention_calls = 0
    try:
        for cycle in range(cycles):
            state = runtime.create_request(96)
            # Active Tier admission is request-wide, while append work is
            # intentionally bounded to the writable hot set.
            _append(runtime, state, 32, value=cycle + 1)
            _append(runtime, state, 32, value=cycle + 1)
            for token in range(2):
                query = torch.full(
                    (1, runtime.num_query_heads, runtime.head_dim),
                    float(token + 1),
                    dtype=runtime.dtype,
                    device=runtime.device,
                )
                result = runtime.attend(
                    (state,), 0, query, (1,), phase="decode"
                )
                if not result.provider_metrics.get("prefetch_selected_only", False):
                    raise AssertionError("Active Tier prefetched an unselected page")
                if result.provider_metrics.get("kernel_actual_read_pages") != 2:
                    raise AssertionError("attention read set differs from RGKV budget")
                total_attention_calls += 1
                _append(runtime, state, 1, value=token + 2)
            runtime.quiesce()
            should_sample = (
                completed + 1
            ) % int(sample_every) == 0 or completed + 1 == cycles
            if should_sample:
                snapshots.append(
                    capture_snapshot(
                        runtime, len(snapshots), (completed + 1) * 2
                    )
                )
            runtime.reset(state)
            runtime.page_pool.validate_invariants()
            runtime.release(state)
            runtime.page_pool.validate_invariants()
            completed += 1
            if should_sample:
                snapshots.append(capture_snapshot(runtime, len(snapshots), completed * 2))
        stats = runtime.profile_stats()
        if stats["eviction_count"] <= 0 or stats["prefetch_count"] <= 0:
            raise AssertionError("RGKV Active Tier case did not migrate KV")
        if stats["h2d_kv_bytes"] <= 0 or stats["d2h_kv_bytes"] <= 0:
            raise AssertionError("RGKV Active Tier case did not copy both directions")
        if stats["rgkv_host_authority_page_checks"] != total_attention_calls * 2:
            raise AssertionError(
                "selected PageHandle authority checks do not match the RGKV read set"
            )
        metrics = {
            "completed_rgkv_active_tier_sessions": completed,
            "attention_calls": total_attention_calls,
            "eviction_count": stats["eviction_count"],
            "prefetch_count": stats["prefetch_count"],
            "h2d_kv_bytes": stats["h2d_kv_bytes"],
            "d2h_kv_bytes": stats["d2h_kv_bytes"],
            "rgkv_stale_index_count": stats["rgkv_stale_index_count"],
            "rgkv_cpu_sync_count": stats["rgkv_cpu_sync_count"],
            "rgkv_host_authority_page_checks": stats[
                "rgkv_host_authority_page_checks"
            ],
        }
    finally:
        runtime.close()
    # ActiveTierCoordinator owns a worker pool.  Capture once after close so
    # thread/future closure is part of the stability gate, not hidden by the
    # last in-session sample.
    snapshots.append(capture_snapshot(runtime, len(snapshots), completed * 2))
    return snapshots, metrics


def _cuda_skip_reason(environment):
    if not environment["cuda_available"]:
        visible = environment.get("cuda_visible_devices")
        suffix = (
            " (CUDA_VISIBLE_DEVICES={!r})".format(visible)
            if visible is not None
            else ""
        )
        return "torch.cuda.is_available() is false{}".format(suffix)
    return None


def _real_case_reason():
    model_path = os.environ.get("CASCADE_KV_MODEL_PATH")
    runner = os.environ.get("CASCADE_KV_LONG_STABILITY_RUNNER")
    if not model_path:
        return "CASCADE_KV_MODEL_PATH is not configured"
    if not runner:
        return "CASCADE_KV_LONG_STABILITY_RUNNER is not configured"
    return (
        "external real-model runner integration is declared but not loaded by "
        "this weight-free harness"
    )


def run_validation(
    mode="logic",
    *,
    session_cycles=None,
    decode_lengths=None,
    context_lengths=None,
    lifecycle_cycles=None,
    sample_every=1,
    cuda_device="cuda:0",
):
    mode = str(mode)
    if mode not in {"logic", "cuda-smoke", "full"}:
        raise ValueError("unknown mode {!r}".format(mode))
    environment = capture_environment()
    if session_cycles is None:
        session_cycles = (10, 100, 1000) if mode == "full" else (10,)
    if decode_lengths is None:
        decode_lengths = (32, 128, 1000) if mode == "full" else (32,)
    if context_lengths is None:
        context_lengths = (
            (128, 512, 2048, 8192, 16384) if mode == "full" else (128,)
        )
    cases = []
    if mode in {"logic", "full"}:
        for cycles in session_cycles:
            cases.append(
                _execute_case(
                    "LS-CYCLE-{}".format(cycles),
                    "CPU synthetic session lifecycle x{}".format(cycles),
                    "session_cycle",
                    {"cycles": int(cycles), "decode_tokens_per_cycle": 2, "device": "cpu"},
                    lambda cycles=cycles: run_session_cycles(
                        cycles, 2, "cpu", sample_every=sample_every
                    ),
                )
            )
        for tokens in decode_lengths:
            cases.append(
                _execute_case(
                    "LS-DECODE-{}".format(tokens),
                    "CPU synthetic decode {} tokens".format(tokens),
                    "long_decode",
                    {"tokens": int(tokens), "device": "cpu"},
                    lambda tokens=tokens: run_long_decode(
                        tokens, "cpu", sample_every=sample_every
                    ),
                )
            )
        for tokens in context_lengths:
            cases.append(
                _execute_case(
                    "LS-CONTEXT-{}".format(tokens),
                    "CPU synthetic context {} tokens".format(tokens),
                    "context",
                    {"tokens": int(tokens), "device": "cpu"},
                    lambda tokens=tokens: run_context(tokens, "cpu"),
                )
            )
        lifecycle_cycles = int(
            lifecycle_cycles
            if lifecycle_cycles is not None
            else 1000 if mode == "full" else 10
        )
        if lifecycle_cycles <= 0:
            raise ValueError("lifecycle_cycles must be positive")
        for identifier, name, category, operation in (
            (
                "LS-FORK-COW-ROLLBACK-{}".format(lifecycle_cycles),
                "CPU synthetic fork/COW/rollback x{}".format(lifecycle_cycles),
                "fork_cow_rollback",
                lambda: run_fork_cow_rollback_cycles(lifecycle_cycles),
            ),
            (
                "LS-PREFIX-CYCLE-{}".format(lifecycle_cycles),
                "CPU synthetic Prefix insert/evict x{}".format(lifecycle_cycles),
                "prefix_cycle",
                lambda: run_prefix_cycles(lifecycle_cycles),
            ),
            (
                "LS-RGKV-CYCLE-{}".format(lifecycle_cycles),
                "CPU synthetic RGKV index lifecycle x{}".format(lifecycle_cycles),
                "rgkv_cycle",
                lambda: run_rgkv_index_cycles(lifecycle_cycles),
            ),
        ):
            cases.append(
                _execute_case(
                    identifier,
                    name,
                    category,
                    {"cycles": lifecycle_cycles, "device": "cpu"},
                    operation,
                )
            )
        active_cycles = max(int(item) for item in session_cycles)
        cases.append(
            _execute_case(
                "LS-RGKV-ACTIVE-TIER-{}".format(active_cycles),
                "CPU synthetic RGKV Active Tier session lifecycle x{}".format(
                    active_cycles
                ),
                "rgkv_active_tier_session_cycle",
                {
                    "cycles": active_cycles,
                    "device": "cpu",
                    "gpu_hot_pages": 3,
                    "request_pages": 4,
                    "rgkv_page_budget": 2,
                },
                lambda: run_rgkv_active_tier_session_cycles(
                    active_cycles, sample_every=sample_every
                ),
            )
        )
    if mode in {"cuda-smoke", "full"}:
        reason = _cuda_skip_reason(environment)
        if reason:
            for identifier, name in (
                ("LS-CUDA-CYCLE", "CUDA session-cycle smoke"),
                ("LS-CUDA-DECODE", "CUDA 32-token smoke"),
                ("LS-CUDA-CONTEXT", "CUDA 128-token context smoke"),
            ):
                cases.append(skipped_case(identifier, name, "cuda_smoke", reason))
        else:
            cases.extend(
                (
                    _execute_case(
                        "LS-CUDA-CYCLE",
                        "CUDA session-cycle smoke",
                        "cuda_smoke",
                        {"cycles": 10, "device": cuda_device},
                        lambda: run_session_cycles(10, 2, cuda_device),
                    ),
                    _execute_case(
                        "LS-CUDA-DECODE",
                        "CUDA 32-token smoke",
                        "cuda_smoke",
                        {"tokens": 32, "device": cuda_device},
                        lambda: run_long_decode(32, cuda_device),
                    ),
                    _execute_case(
                        "LS-CUDA-CONTEXT",
                        "CUDA 128-token context smoke",
                        "cuda_smoke",
                        {"tokens": 128, "device": cuda_device},
                        lambda: run_context(128, cuda_device),
                    ),
                )
            )
    if mode == "full":
        reason = _real_case_reason()
        for identifier, name in (
            ("LS-REAL-100-SESSION", "Real-model 100 session lifecycle"),
            ("LS-REAL-1000-DECODE", "Real-model 1000-token generation"),
            ("LS-REAL-8K", "Real-model 8K context"),
            ("LS-REAL-16K", "Real-model 16K context"),
        ):
            cases.append(skipped_case(identifier, name, "real_model", reason))
    counts = {
        status: sum(item["status"] == status for item in cases)
        for status in (PASS, FAIL, SKIPPED, BLOCKED)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "mode": mode,
        "status": FAIL if counts[FAIL] else BLOCKED if counts[BLOCKED] else PASS,
        "qualification": NOT_QUALIFIED,
        "qualification_reasons": [
            "synthetic logic and shared-environment CUDA smoke are not production qualification",
            "exclusive CUDA fault injection and real-model stability gates remain",
        ],
        "counts": counts,
        "environment": environment,
        "matrix": {
            "session_cycles": [int(item) for item in session_cycles],
            "decode_lengths": [int(item) for item in decode_lengths],
            "context_lengths": [int(item) for item in context_lengths],
            "lifecycle_cycles": int(
                lifecycle_cycles
                if lifecycle_cycles is not None
                else 1000 if mode == "full" else 10
            ),
        },
        "fault_injection_catalog": [
            "append_layer_n_failure",
            "cow_failure",
            "attention_failure",
            "fence_query_failure",
            "cancel_during_operation",
            "operation_timeout",
        ],
        "cases": cases,
    }


def render_markdown(report):
    env = report["environment"]
    nvidia = env.get("nvidia_smi", {})
    lines = [
        "# KV Long-Stability Report",
        "",
        "- Mode: `{}`".format(report["mode"]),
        "- Execution status: `{}`".format(report["status"]),
        "- Qualification: `{}`".format(report["qualification"]),
        "- Cases: PASS={}, FAIL={}, SKIPPED_WITH_REASON={}, BLOCKED={}".format(
            report["counts"][PASS], report["counts"][FAIL],
            report["counts"][SKIPPED], report["counts"].get(BLOCKED, 0),
        ),
        "- CUDA available: `{}`".format(env["cuda_available"]),
        "- CUDA_VISIBLE_DEVICES: `{}`".format(
            env.get("cuda_visible_devices", "unset")
        ),
        "- No external GPU process at capture: `{}`".format(
            nvidia.get("exclusive_at_capture", "unknown")
        ),
        "",
        "> This report is logic/smoke evidence. It does not qualify CUDA COW, "
        "Attention Fence, 70B generation, RGKV performance, or Tiered CUDA KV.",
        "> A process snapshot is not an exclusive reservation; CUDA was disabled "
        "for this logic-only synthetic report.",
        "",
        "## Cases",
        "",
        "| Case | Category | Status | Duration (s) | Reason |",
        "|---|---|---:|---:|---|",
    ]
    for case in report["cases"]:
        reason = (case.get("reason") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| `{}` | {} | `{}` | {:.3f} | {} |".format(
                case["case_id"], case["category"], case["status"],
                case["duration_seconds"], reason,
            )
        )
    lines.extend(
        [
            "",
            "## Qualification blockers",
            "",
        ]
    )
    lines.extend("- " + item for item in report["qualification_reasons"])
    lines.extend(
        [
            "",
            "## Fault-injection hooks reserved",
            "",
        ]
    )
    lines.extend("- `" + item + "`" for item in report["fault_injection_catalog"])
    lines.append("")
    return "\n".join(lines)


def write_reports(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_document = {
        "schema_version": report["schema_version"],
        "generated_at": report["generated_at"],
        "mode": report["mode"],
        "cases": report["cases"],
    }
    summary = dict(report)
    summary.pop("cases", None)
    environment = summary.pop("environment", {})
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "cases.json").write_text(
        json.dumps(cases_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return {
        "environment": str(output_dir / "environment.json"),
        "summary": str(output_dir / "summary.json"),
        "cases": str(output_dir / "cases.json"),
        "report": str(output_dir / "report.md"),
    }


def _parse_ints(value):
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", "--profile", dest="mode",
        choices=("logic", "cuda-smoke", "full"), default="logic",
    )
    parser.add_argument("--session-cycles", type=_parse_ints)
    parser.add_argument("--decode-lengths", type=_parse_ints)
    parser.add_argument("--context-lengths", type=_parse_ints)
    parser.add_argument("--lifecycle-cycles", type=int)
    parser.add_argument(
        "--sample-every",
        type=int,
        help="snapshot interval; defaults to 100 for full and 1 otherwise",
    )
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument(
        "--output-dir", default=str(ROOT / "reports" / "kv_long_stability")
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.sample_every is None:
        args.sample_every = 100 if args.mode == "full" else 1
    if args.sample_every <= 0:
        raise SystemExit("--sample-every must be positive")
    report = run_validation(
        args.mode,
        session_cycles=args.session_cycles,
        decode_lengths=args.decode_lengths,
        context_lengths=args.context_lengths,
        lifecycle_cycles=args.lifecycle_cycles,
        sample_every=args.sample_every,
        cuda_device=args.cuda_device,
    )
    paths = write_reports(report, args.output_dir)
    print(json.dumps({"status": report["status"], "qualification": report["qualification"], "counts": report["counts"], "artifacts": paths}, indent=2, sort_keys=True))
    return 1 if report["status"] in {FAIL, BLOCKED} else 0


if __name__ == "__main__":
    raise SystemExit(main())
