#!/usr/bin/env python3
"""Qualification harness for asynchronous CUDA KV lifecycle contracts.

``logic`` executes CPU-only fault injection for the shared Fence contract.
``cuda-smoke`` may run on a shared GPU, but every result is labelled
``SMOKE_ONLY``.  ``qualification`` refuses to run when the selected physical
GPU has another compute process.  A refusal is reported as
``BLOCKED_NOT_EXCLUSIVE`` and is never converted into a PASS.

This tool validates synthetic KV lifecycle behavior.  It does not qualify a
model, Prefill performance, or 70B generation.
"""

import argparse
from concurrent.futures import Future
import json
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from layer_streaming.attention.paged import (
    ReferencePagedExactBackend,
    default_paged_registry,
)
from layer_streaming.kv import (
    KVLifecycleError,
    KVOperationFence,
    KVPagePoolV1,
    PagedKVRuntime,
    TorchPagedKVKernelBackend,
)
from layer_streaming.kv.stores import GPUKVStore
from layer_streaming.kv.ownership import OwnershipManager
from layer_streaming.kv_policy import KVDataType, KVPolicy
from layer_streaming.providers.base import PagedProviderBundle
from layer_streaming.capability_state import CapabilityState
from tools.qualification_common import (
    BLOCKED,
    BLOCKED_NOT_EXCLUSIVE,
    FAIL,
    PASS,
    SKIPPED,
    capture_cuda_environment,
    qualification_admission,
    utc_now,
    write_report_bundle,
)


SCHEMA_VERSION = 1


CASE_NAMES = {
    "B1": "COW copy cross-stream Fence and pin lifecycle",
    "B2": "Attention Fence release/reset cross-stream lifecycle",
    "B3": "Append Layer-N failure quiesce and rollback",
    "B4-QUERY": "Fence Event query error propagation",
    "B4-SYNC": "Fence Event synchronize error propagation",
    "B4-TIMEOUT": "Fence bounded timeout",
    "B4-CANCEL": "Fence cancel and double cancel",
    "B4-DRAIN": "Attention Fence double drain",
}


def make_case(case_id, status, evidence, *, reason=None, metrics=None,
              duration_seconds=0.0, error_traceback=None):
    return {
        "case_id": case_id,
        "name": CASE_NAMES[case_id],
        "status": status,
        "evidence": evidence,
        "reason": reason,
        "duration_seconds": float(duration_seconds),
        "metrics": dict(metrics or {}),
        "traceback": error_traceback,
    }


def execute(case_id, evidence, function):
    started = time.perf_counter()
    try:
        metrics = function() or {}
        return make_case(
            case_id, PASS, evidence, metrics=metrics,
            duration_seconds=time.perf_counter() - started,
        )
    except BaseException as error:
        return make_case(
            case_id, FAIL, evidence,
            reason="{}: {}".format(type(error).__name__, error),
            duration_seconds=time.perf_counter() - started,
            error_traceback=traceback.format_exc(),
        )


class PendingEvent:
    def __init__(self, done=False):
        self.done = bool(done)

    def query(self):
        return self.done


class QueryFailureEvent:
    def query(self):
        raise RuntimeError("injected event query error")


class SynchronizeFailureEvent:
    def query(self):
        return True

    def synchronize(self):
        raise RuntimeError("injected event synchronize error")


def _logic_cow_contract():
    pool = KVPagePoolV1(2, "logic-cow", "bf16")
    source_mapping = ("test", "source")
    target_mapping = ("test", "target")
    source = pool.allocate(owner_hint=1, logical_mapping=source_mapping)
    target = pool.allocate(owner_hint=1, logical_mapping=target_mapping)
    pool.activate(source, 3)
    pool.seal(source, 3)
    pool.begin_copy(source, target)
    source_pending = pool.descriptor(source)
    target_pending = pool.descriptor(target)
    assert target_pending.state.value == "copying"
    assert source_pending.pin_count == source_pending.inflight_io == 1
    assert target_pending.pin_count == target_pending.inflight_io == 1
    pool.end_copy(source, target, 3)
    assert pool.descriptor(source).pin_count == 0
    assert pool.descriptor(target).pin_count == 0
    pool.release(target, logical_mapping=target_mapping)
    pool.release(source, logical_mapping=source_mapping)
    assert pool.validate_invariants()
    return {"source_target_pinned_while_copying": True, "final_free_pages": 2}


def _make_runtime(device="cpu", layer_count=1, backend=None):
    device = torch.device(device)
    name = "cuda_async_harness"
    registry = default_paged_registry(load_cuda=False)
    registry.register(
        PagedProviderBundle(
            name=name,
            attention_backend=ReferencePagedExactBackend(),
            kv_kernel_backend=backend or TorchPagedKVKernelBackend(),
        )
    )
    dtype = torch.bfloat16
    return PagedKVRuntime(
        layer_count=layer_count,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
        page_count=8,
        page_size=16,
        dtype=dtype,
        device=device,
        policy=KVPolicy(
            dtype=KVDataType.BF16,
            page_size=16,
            attention_backend=name,
        ),
        provider_registry=registry,
        allow_reference=True,
    )


def _logic_append_failure():
    class FailThirdAppend(TorchPagedKVKernelBackend):
        def __init__(self):
            self.calls = 0

        def append_kv(self, append_input):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("injected append layer 2 failure")
            return super().append_kv(append_input)

    runtime = _make_runtime("cpu", layer_count=3, backend=FailThirdAppend())
    try:
        state = runtime.create_request(32, request_id=103)
        value = torch.ones(3, 1, 4, dtype=runtime.dtype)
        runtime.append((state,), 0, value, value, (3,))
        runtime.append((state,), 1, value, value, (3,))
        try:
            runtime.append((state,), 2, value, value, (3,))
            raise AssertionError("injected Layer-N failure did not fire")
        except RuntimeError as error:
            assert "layer 2" in str(error)
        profile = runtime.page_pool.profile()
        assert state.pending_append is None
        assert state.sequence_length == 0
        assert profile["allocated_pages"] == 0
        assert profile["total_pin_count"] == 0
        assert profile["inflight_compute"] == 0
        assert profile["inflight_io"] == 0
        assert runtime.page_pool.validate_invariants()
        return {
            "failed_layer": 2,
            "pending_append": 0,
            "final_allocated_pages": 0,
            "final_pin_count": 0,
            "original_error_preserved": True,
        }
    finally:
        runtime.close()


def _fence_query_error():
    fence = KVOperationFence(
        operation_id="logic-query-error", request_id=1, kind="test",
        cuda_event=QueryFailureEvent(),
    )
    try:
        fence.wait(timeout_seconds=0.01)
        raise AssertionError("event query error was swallowed")
    except KVLifecycleError as error:
        assert isinstance(error.__cause__, RuntimeError)
        assert "query" in str(error.__cause__)
    assert fence.status == "failed"
    try:
        OwnershipManager._wait_event(QueryFailureEvent(), 0.01, "injected")
        raise AssertionError("ownership Event query error was swallowed")
    except RuntimeError as error:
        assert "query" in str(error)
    return {
        "query_error_propagated": True,
        "ownership_query_error_propagated": True,
        "status": fence.status,
    }


def _fence_synchronize_error():
    fence = KVOperationFence(
        operation_id="logic-synchronize-error", request_id=1, kind="test",
        cuda_event=SynchronizeFailureEvent(),
    )
    try:
        fence.wait(timeout_seconds=0.01)
        raise AssertionError("event synchronize error was swallowed")
    except KVLifecycleError as error:
        assert isinstance(error.__cause__, RuntimeError)
        assert "synchronize" in str(error.__cause__)
    assert fence.status == "failed"
    try:
        OwnershipManager._wait_event(
            SynchronizeFailureEvent(), 0.01, "injected"
        )
        raise AssertionError("ownership Event synchronize error was swallowed")
    except RuntimeError as error:
        assert "synchronize" in str(error)
    return {
        "synchronize_error_propagated": True,
        "ownership_synchronize_error_propagated": True,
        "status": fence.status,
    }


def _fence_timeout():
    fence = KVOperationFence(
        operation_id="logic-timeout", request_id=1, kind="test",
        cuda_event=PendingEvent(False),
    )
    started = time.perf_counter()
    try:
        fence.wait(timeout_seconds=0.003)
        raise AssertionError("pending fence did not time out")
    except KVLifecycleError as error:
        assert "timed out" in str(error)
    runtime = _make_runtime("cpu")
    try:
        state = runtime.create_request(16, request_id=105)
        value = torch.ones(1, 1, 4, dtype=runtime.dtype)
        runtime.append((state,), 0, value, value, (1,))
        handle = state.block_table.handles[0]
        event = PendingEvent(False)
        runtime.page_pool.pin(handle)
        attention = runtime.ownership._fence(
            "attention_layer_0", request_id=(state.request_id,),
            source_handles=(handle,), cuda_event=event,
        )
        runtime.ownership.attention_pins[0] = [handle]
        runtime.ownership.attention_fences[0] = attention
        runtime.ownership._attention_fence_layers[attention.operation_id] = 0
        runtime.ownership._remember_fence(attention)
        try:
            runtime.drain_attention_fence(
                attention.operation_id, timeout_seconds=0.003
            )
            raise AssertionError("pending Attention Fence did not time out")
        except KVLifecycleError as error:
            assert "timed out" in str(error)
        assert runtime.page_pool.descriptor(handle).pin_count == 1
        event.done = True
        runtime.drain_attention_fence(attention.operation_id)
        assert runtime.page_pool.descriptor(handle).pin_count == 0
        runtime.release(state)
    finally:
        runtime.close()
    return {
        "timeout_raised": True,
        "elapsed_seconds": time.perf_counter() - started,
        "status": fence.status,
        "pin_retained_until_late_completion": True,
        "cleanup_after_completion": True,
    }


def _fence_double_cancel():
    future = Future()
    fence = KVOperationFence(
        operation_id="logic-cancel", request_id=1, kind="test",
        io_future=future,
    )
    fence.cancel()
    fence.cancel()
    assert fence.cancelled
    assert future.cancelled()
    try:
        fence.wait(timeout_seconds=0.01)
        raise AssertionError("cancelled fence wait was accepted")
    except KVLifecycleError as error:
        assert "cancelled" in str(error)
    return {"double_cancel_idempotent": True, "future_cancelled": True}


def _attention_double_drain(device="cpu"):
    runtime = _make_runtime(device)
    try:
        state = runtime.create_request(32, request_id=104)
        key = torch.ones(
            1, 1, 4, dtype=runtime.dtype, device=runtime.device
        )
        runtime.append((state,), 0, key, key, (1,))
        query = torch.ones(
            1, 2, 4, dtype=runtime.dtype, device=runtime.device
        )
        result = runtime.attend((state,), 0, query, (1,), phase="decode")
        fence_id = result.provider_metrics["attention_fence_id"]
        first = runtime.drain_attention_fence(fence_id)
        second = runtime.drain_attention_fence(fence_id)
        assert first is second
        descriptor = runtime.page_pool.descriptor(state.block_table.handles[0])
        assert descriptor.pin_count == 0
        assert descriptor.inflight_compute == 0
        runtime.release(state)
        assert runtime.page_pool.free_pages == runtime.page_pool.page_count
        return {
            "fence_id": fence_id,
            "double_drain_idempotent": True,
            "final_pin_count": 0,
        }
    finally:
        runtime.close()


def _cuda_cow_cross_stream(device):
    device = torch.device(device)
    pool = KVPagePoolV1(2, "cuda-cow", "bf16")
    store = GPUKVStore(1, 2, 1, 16, 4, torch.bfloat16, device)
    source_mapping = ("test", "source")
    target_mapping = ("test", "target")
    source = pool.allocate(owner_hint=1, logical_mapping=source_mapping)
    target = pool.allocate(owner_hint=1, logical_mapping=target_mapping)
    copy_stream = torch.cuda.Stream(device=device)
    use_stream = torch.cuda.Stream(device=device)
    metrics = {}
    try:
        source_payload = torch.arange(
            64, dtype=torch.float32, device=device
        ).reshape(1, 16, 4).to(torch.bfloat16)
        store.keys[0, source.page_id].copy_(source_payload)
        store.values[0, source.page_id].copy_(source_payload + 1)
        torch.cuda.synchronize(device)
        pool.activate(source, 16)
        pool.seal(source, 16)
        pool.begin_copy(source, target)
        with torch.cuda.stream(copy_stream):
            store.copy_page(source.page_id, target.page_id, 16)
            copy_event = torch.cuda.Event(enable_timing=False)
            copy_event.record(copy_stream)
        fence = KVOperationFence(
            operation_id="cuda-cow-copy", request_id=1, kind="cow_copy",
            source_handles=(source,), target_handles=(target,),
            cuda_event=copy_event,
        )
        pending = pool.descriptor(target)
        assert pending.state.value == "copying"
        assert pending.pin_count == pending.inflight_io == 1
        with torch.cuda.stream(use_stream):
            use_stream.wait_event(copy_event)
            observed = store.keys[0, target.page_id].float().sum()
        fence.wait(timeout_seconds=10.0)
        pool.end_copy(source, target, 16)
        use_stream.synchronize()
        expected = float(source_payload.float().sum().item())
        actual = float(observed.item())
        assert actual == expected, (actual, expected)
        assert pool.descriptor(source).pin_count == 0
        assert pool.descriptor(target).pin_count == 0
        pool.release(target, logical_mapping=target_mapping)
        pool.release(source, logical_mapping=source_mapping)
        assert pool.validate_invariants()
        metrics.update({
            "copy_stream": int(copy_stream.cuda_stream),
            "use_stream": int(use_stream.cuda_stream),
            "target_copying_before_fence": True,
            "dependent_read_matches": True,
            "final_pin_count": 0,
        })
    finally:
        store.close()
    runtime = _make_runtime(device)
    try:
        parent = runtime.create_request(32, request_id=301)
        value = torch.arange(
            12, dtype=torch.float32, device=runtime.device
        ).reshape(3, 1, 4).to(runtime.dtype)
        runtime.append((parent,), 0, value, value + 1, (3,))
        source = parent.block_table.handles[0]
        child = runtime.fork(parent, request_id=302)
        with torch.cuda.stream(copy_stream):
            runtime.append(
                (child,), 0, value[:1] + 2, value[:1] + 3, (1,)
            )
        target = child.block_table.handles[0]
        assert target.identity() != source.identity()
        assert runtime.page_pool.descriptor(source).pin_count == 0
        assert runtime.page_pool.descriptor(target).pin_count == 0
        with torch.cuda.stream(use_stream):
            query = torch.ones(
                1, 2, 4, dtype=runtime.dtype, device=runtime.device
            )
            result = runtime.attend(
                (child,), 0, query, (1,), phase="decode"
            )
        runtime.drain_attention_fence(
            result.provider_metrics["attention_fence_id"],
            timeout_seconds=10.0,
        )
        runtime.release(child)
        runtime.release(parent)
        assert runtime.page_pool.free_pages == runtime.page_pool.page_count
        metrics.update(
            {
                "ownership_cow_source_page": source.page_id,
                "ownership_cow_target_page": target.page_id,
                "target_published_after_copy_fence": True,
                "ownership_final_free_pages": runtime.page_pool.free_pages,
            }
        )
    finally:
        runtime.close()
    return metrics


def _cuda_attention_release_reset(device):
    runtime = _make_runtime(device)
    outcomes = {}
    try:
        stream = torch.cuda.Stream(device=runtime.device)
        for operation, request_id in (("release", 201), ("reset", 202)):
            state = runtime.create_request(32, request_id=request_id)
            key = torch.ones(
                1, 1, 4, dtype=runtime.dtype, device=runtime.device
            )
            runtime.append((state,), 0, key, key, (1,))
            query = torch.ones(
                1, 2, 4, dtype=runtime.dtype, device=runtime.device
            )
            with torch.cuda.stream(stream):
                result = runtime.attend(
                    (state,), 0, query, (1,), phase="decode"
                )
            fence_id = result.provider_metrics["attention_fence_id"]
            if operation == "release":
                runtime.release(state)
            else:
                runtime.reset(state)
                runtime.release(state)
            profile = runtime.page_pool.profile()
            assert profile["total_pin_count"] == 0
            assert profile["inflight_compute"] == 0
            outcomes[operation] = {
                "fence_id": fence_id,
                "final_pin_count": profile["total_pin_count"],
            }
        assert runtime.page_pool.free_pages == runtime.page_pool.page_count
        return {
            "operations": outcomes,
            "final_free_pages": runtime.page_pool.free_pages,
        }
    finally:
        runtime.close()


def _cuda_append_failure(device):
    class DelayedFailThirdAppend(TorchPagedKVKernelBackend):
        def __init__(self):
            self.calls = 0

        def append_kv(self, append_input):
            self.calls += 1
            if self.calls == 1 and hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(2_000_000)
            if self.calls == 3:
                raise RuntimeError("injected CUDA append layer 2 failure")
            return super().append_kv(append_input)

    runtime = _make_runtime(device, 3, DelayedFailThirdAppend())
    try:
        state = runtime.create_request(32, request_id=203)
        value = torch.ones(
            3, 1, 4, dtype=runtime.dtype, device=runtime.device
        )
        stream = torch.cuda.Stream(device=runtime.device)
        with torch.cuda.stream(stream):
            runtime.append((state,), 0, value, value, (3,))
            runtime.append((state,), 1, value, value, (3,))
        try:
            runtime.append((state,), 2, value, value, (3,))
            raise AssertionError("injected CUDA Layer-N failure did not fire")
        except RuntimeError as error:
            assert "layer 2" in str(error)
        profile = runtime.page_pool.profile()
        assert state.pending_append is None
        assert state.sequence_length == 0
        assert profile["allocated_pages"] == 0
        assert profile["total_pin_count"] == 0
        assert profile["inflight_compute"] == 0
        assert profile["inflight_io"] == 0
        assert runtime.page_pool.validate_invariants()
        return {
            "failed_layer": 2,
            "pending_append": 0,
            "final_allocated_pages": 0,
            "final_pin_count": 0,
            "original_error_preserved": True,
        }
    finally:
        runtime.close()


def _logic_cases():
    evidence = CapabilityState.LOGIC_VALIDATED.value
    return [
        execute("B1", evidence, _logic_cow_contract),
        make_case(
            "B2", SKIPPED, evidence,
            reason="cross-stream Attention release/reset requires CUDA",
        ),
        execute("B3", evidence, _logic_append_failure),
        execute("B4-QUERY", evidence, _fence_query_error),
        execute("B4-SYNC", evidence, _fence_synchronize_error),
        execute("B4-TIMEOUT", evidence, _fence_timeout),
        execute("B4-CANCEL", evidence, _fence_double_cancel),
        execute("B4-DRAIN", evidence, lambda: _attention_double_drain("cpu")),
    ]


def _cuda_cases(device, evidence, qualification):
    del qualification
    return [
        execute("B1", evidence, lambda: _cuda_cow_cross_stream(device)),
        execute("B2", evidence, lambda: _cuda_attention_release_reset(device)),
        execute("B3", evidence, lambda: _cuda_append_failure(device)),
        execute("B4-QUERY", evidence, _fence_query_error),
        execute("B4-SYNC", evidence, _fence_synchronize_error),
        execute("B4-TIMEOUT", evidence, _fence_timeout),
        execute("B4-CANCEL", evidence, _fence_double_cancel),
        execute("B4-DRAIN", evidence, lambda: _attention_double_drain(device)),
    ]


def _unavailable_cases(status, evidence, reason):
    return [
        make_case(case_id, status, evidence, reason=reason)
        for case_id in CASE_NAMES
    ]


def _compact_gpu_snapshot(environment):
    return {
        "captured_at": environment.get("captured_at"),
        "selected_physical_gpu": environment.get("selected_physical_gpu"),
        "selected_gpu_compute_processes": environment.get(
            "selected_gpu_compute_processes", []
        ),
        "selected_gpu_external_compute_processes": environment.get(
            "selected_gpu_external_compute_processes", 0
        ),
        "exclusive_snapshot": environment.get("exclusive_snapshot", False),
        "reservation_evidence_present": environment.get(
            "reservation_evidence_present", False
        ),
        "reservation_evidence": environment.get("reservation_evidence"),
    }


def run_validation(mode="logic", device="cuda:0"):
    if mode not in {"logic", "cuda-smoke", "qualification"}:
        raise ValueError("unknown mode {!r}".format(mode))
    environment = capture_cuda_environment(ROOT, device)
    if mode == "logic":
        cases = _logic_cases()
        qualification = CapabilityState.LOGIC_VALIDATED.value
    elif not environment["cuda_available"]:
        reason = "torch.cuda.is_available() is false"
        cases = _unavailable_cases(
            SKIPPED, CapabilityState.CUDA_SMOKE.value, reason
        )
        qualification = CapabilityState.QUALIFICATION_READY.value
    elif environment.get("selected_physical_gpu") is None:
        reason = "selected physical GPU could not be resolved: {}".format(
            environment.get("nvidia_smi_errors")
        )
        status = BLOCKED if mode == "qualification" else SKIPPED
        evidence = (
            CapabilityState.QUALIFICATION_READY.value
            if mode == "qualification"
            else CapabilityState.CUDA_SMOKE.value
        )
        cases = _unavailable_cases(status, evidence, reason)
        qualification = CapabilityState.QUALIFICATION_READY.value
    elif mode == "qualification":
        admitted, _, reason = qualification_admission(
            environment, require_reservation=True
        )
        if not admitted:
            cases = _unavailable_cases(
                BLOCKED, CapabilityState.QUALIFICATION_READY.value, reason
            )
            qualification = CapabilityState.QUALIFICATION_READY.value
        else:
            cases = _cuda_cases(
                device, CapabilityState.QUALIFIED.value, True
            )
            qualification = CapabilityState.QUALIFIED.value
    else:
        evidence = (
            CapabilityState.QUALIFIED.value
            if mode == "qualification"
            else CapabilityState.CUDA_SMOKE.value
        )
        cases = _cuda_cases(device, evidence, mode == "qualification")
        qualification = (
            CapabilityState.QUALIFIED.value
            if mode == "qualification"
            else CapabilityState.CUDA_SMOKE.value
        )
    if mode == "qualification" and any(
        case["status"] == PASS for case in cases
    ):
        end_environment = capture_cuda_environment(ROOT, device)
        environment["end_snapshot"] = _compact_gpu_snapshot(end_environment)
        start_gpu = environment.get("selected_physical_gpu") or {}
        end_gpu = end_environment.get("selected_physical_gpu") or {}
        start_job = (environment.get("reservation_evidence") or {}).get("job_id")
        end_job = (end_environment.get("reservation_evidence") or {}).get("job_id")
        invalid_reason = None
        if int(end_environment.get("selected_gpu_external_compute_processes", 0)):
            invalid_reason = "INVALIDATED_EXTERNAL_GPU_ACTIVITY"
        elif start_gpu.get("uuid") != end_gpu.get("uuid"):
            invalid_reason = "INVALIDATED_GPU_IDENTITY_CHANGED"
        elif not end_environment.get("reservation_evidence_present", False):
            invalid_reason = "INVALIDATED_RESERVATION_EVIDENCE_LOST"
        elif start_job != end_job:
            invalid_reason = "INVALIDATED_RESERVATION_ID_CHANGED"
        if invalid_reason is not None:
            for case in cases:
                if case["status"] == PASS:
                    case["status"] = FAIL
                    case["reason"] = invalid_reason
            qualification = CapabilityState.QUALIFICATION_READY.value
    counts = {
        status: sum(case["status"] == status for case in cases)
        for status in (PASS, FAIL, SKIPPED, BLOCKED)
    }
    overall = FAIL if counts[FAIL] else BLOCKED if counts[BLOCKED] else PASS
    if mode == "qualification" and (counts[FAIL] or counts[BLOCKED] or counts[SKIPPED]):
        qualification = CapabilityState.QUALIFICATION_READY.value
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "mode": mode,
        "status": overall,
        "qualification": qualification,
        "qualification_scope": "synthetic_dense_gpu_kv_async_lifecycle",
        "environment": environment,
        "counts": counts,
        "cases": cases,
    }


def render_markdown(report):
    environment = report["environment"]
    lines = [
        "# KV CUDA Async Validation",
        "",
        "- Mode: `{}`".format(report["mode"]),
        "- Status: `{}`".format(report["status"]),
        "- Qualification: `{}`".format(report["qualification"]),
        "- Selected GPU exclusive at snapshot: `{}`".format(
            environment.get("exclusive_snapshot")
        ),
        "- Scheduler reservation evidence: `{}`".format(
            environment.get("reservation_evidence_present", False)
        ),
        "- PASS={}, FAIL={}, SKIPPED_WITH_REASON={}, BLOCKED={}".format(
            report["counts"][PASS], report["counts"][FAIL],
            report["counts"][SKIPPED], report["counts"][BLOCKED],
        ),
        "",
        "> `cuda-smoke` is never qualification evidence. Qualification mode "
        "requires scheduler-backed GPU allocation evidence in addition to "
        "clean process snapshots. This harness does not qualify 70B or Prefill "
        "performance.",
        "",
        "| Case | Status | Evidence | Reason |",
        "|---|---:|---|---|",
    ]
    for case in report["cases"]:
        reason = str(case.get("reason") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| `{}` {} | `{}` | `{}` | {} |".format(
                case["case_id"], case["name"], case["status"],
                case["evidence"], reason,
            )
        )
    lines.extend(
        [
            "",
            "## Fence error coverage",
            "",
            "Both Event `query()` and final `synchronize()` failures are "
            "injected through the real `KVOperationFence.wait()` path. "
            "Timeout, cancel, double cancel and double drain have separate "
            "case results.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report, output_dir):
    summary = dict(report)
    summary.pop("cases", None)
    summary.pop("environment", None)
    return write_report_bundle(
        output_dir, report["environment"], report["cases"], summary,
        render_markdown(report),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("logic", "cuda-smoke", "qualification"),
        default="logic",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", default=str(ROOT / "reports" / "kv_cuda_async")
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_validation(args.mode, args.device)
    artifacts = write_reports(report, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "qualification": report["qualification"],
        "counts": report["counts"],
        "artifacts": artifacts,
    }, indent=2, sort_keys=True))
    return 1 if report["status"] == FAIL else 2 if report["status"] == BLOCKED else 0


if __name__ == "__main__":
    raise SystemExit(main())
