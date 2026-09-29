#!/usr/bin/env python3
"""Validate Active GPU/Pinned-CPU Tier logic, smoke, or qualification entry."""

import argparse
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming.kv.runtime import PagedKVRuntime  # noqa: E402
from layer_streaming.kv.api import RequestKVCacheV1  # noqa: E402
from layer_streaming.kv_policy import KVPolicy  # noqa: E402
from layer_streaming.generation_session import GenerationSession  # noqa: E402
from layer_streaming.chat import SamplingConfig  # noqa: E402
from tools.qualification_common import (  # noqa: E402
    BLOCKED,
    capture_cuda_environment,
    qualification_admission,
    utc_now,
    write_report_bundle,
)


CASE_MATRIX = ((2, 6, 16), (4, 16, 16), (4, 8, 32))
SESSION_CASE_ID = "ACTIVE-TIER-GENERATION-SESSION"


def _cpu_factory(shape, **kwargs):
    return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


def _percentile(values, fraction):
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return None
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _policy(
    *, tiered, cuda, page_size, logical_page_bytes, hot_pages, logical_pages
):
    return KVPolicy(
        storage="gpu_cpu" if tiered else "gpu",
        dtype="fp16",
        selection="none",
        reuse="request_only",
        attention_backend=("generic_cuda" if cuda else "reference_paged_exact"),
        page_size=page_size,
        gpu_hot_budget_bytes=(hot_pages * logical_page_bytes if tiered else 0),
        cpu_budget_bytes=(logical_pages * logical_page_bytes if tiered else 0),
        gpu_migration_slots_bytes=(logical_page_bytes if tiered else 0),
        cpu_migration_slots_bytes=(logical_page_bytes if tiered else 0),
        gpu_high_watermark_bytes=(hot_pages * logical_page_bytes if tiered else 0),
        gpu_low_watermark_bytes=((hot_pages - 1) * logical_page_bytes if tiered else 0),
    )


def _runtime(device, *, tiered, hot_pages, logical_pages, page_size):
    layers, query_heads, kv_heads, head_dim = 2, 4, 2, 8
    dtype = torch.float16
    cuda = torch.device(device).type == "cuda"
    logical_page_bytes = (
        2 * layers * kv_heads * page_size * head_dim
        * torch.empty((), dtype=dtype).element_size()
    )
    return PagedKVRuntime(
        layer_count=layers,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        page_count=logical_pages,
        page_size=page_size,
        dtype=dtype,
        device=device,
        policy=_policy(
            tiered=tiered,
            cuda=cuda,
            page_size=page_size,
            logical_page_bytes=logical_page_bytes,
            hot_pages=hot_pages,
            logical_pages=logical_pages,
        ),
        allow_reference=(torch.device(device).type == "cpu"),
        prefetch_timeout_seconds=10.0,
        tier_tensor_factory=(
            _cpu_factory if tiered and torch.device(device).type == "cpu" else None
        ),
    )


def _payload(device, dtype, layer, start, count, kv_heads=2, head_dim=8):
    values = torch.arange(
        start * kv_heads * head_dim,
        (start + count) * kv_heads * head_dim,
        dtype=torch.float32,
        device=device,
    ).reshape(count, kv_heads, head_dim)
    key = values.mul(0.0001).add(float(layer + 1)).to(dtype)
    value = values.mul(0.0002).add(float(layer + 10)).to(dtype)
    return key, value


def _query(device, dtype, layer, position, query_heads=4, head_dim=8):
    return (
        torch.arange(query_heads * head_dim, dtype=torch.float32, device=device)
        .reshape(1, query_heads, head_dim)
        .mul(0.001)
        .add(float(layer + 1 + position * 0.00001))
        .to(dtype)
    )


def run_case(device, *, hot_pages, logical_pages, page_size):
    if hot_pages < 2 or logical_pages <= hot_pages:
        raise ValueError("case requires 2 <= hot_pages < logical_pages")
    torch.manual_seed(20260809)
    device_value = torch.device(device)
    total_tokens = logical_pages * page_size
    chunk_tokens = (hot_pages - 1) * page_size

    if device_value.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_value)
    dense_baseline_allocated = (
        int(torch.cuda.memory_allocated(device_value))
        if device_value.type == "cuda"
        else None
    )
    dense_baseline_reserved = (
        int(torch.cuda.memory_reserved(device_value))
        if device_value.type == "cuda"
        else None
    )
    dense = _runtime(
        device,
        tiered=False,
        hot_pages=logical_pages,
        logical_pages=logical_pages,
        page_size=page_size,
    )
    dense_state = dense.create_request(total_tokens)
    dense_outputs = []
    dense_latencies = []
    start = 0
    try:
        while start < total_tokens:
            count = min(chunk_tokens, total_tokens - start)
            began = time.perf_counter()
            for layer in range(2):
                key, value = _payload(
                    device_value, torch.float16, layer, start, count
                )
                dense.append((dense_state,), layer, key, value, (count,))
                query = _query(
                    device_value, torch.float16, layer, start + count - 1
                )
                output = dense.attend(
                    (dense_state,), layer, query, (1,), phase="decode"
                ).output
                if device_value.type == "cuda":
                    torch.cuda.synchronize(device_value)
                dense_outputs.append(output.detach().cpu())
            dense_latencies.append((time.perf_counter() - began) * 1000.0)
            start += count
        dense_peak_allocated = (
            int(torch.cuda.max_memory_allocated(device_value))
            if device_value.type == "cuda"
            else None
        )
        dense_peak_reserved = (
            int(torch.cuda.max_memory_reserved(device_value))
            if device_value.type == "cuda"
            else None
        )
    finally:
        dense.close()
    if device_value.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_value)
    tiered_baseline_allocated = (
        int(torch.cuda.memory_allocated(device_value))
        if device_value.type == "cuda"
        else None
    )
    tiered_baseline_reserved = (
        int(torch.cuda.memory_reserved(device_value))
        if device_value.type == "cuda"
        else None
    )

    tiered = _runtime(
        device,
        tiered=True,
        hot_pages=hot_pages,
        logical_pages=logical_pages,
        page_size=page_size,
    )
    tiered_state = tiered.create_request(total_tokens)
    tiered_latencies = []
    max_error = 0.0
    output_index = 0
    start = 0
    try:
        while start < total_tokens:
            count = min(chunk_tokens, total_tokens - start)
            began = time.perf_counter()
            for layer in range(2):
                key, value = _payload(
                    device_value, torch.float16, layer, start, count
                )
                tiered.append((tiered_state,), layer, key, value, (count,))
                query = _query(
                    device_value, torch.float16, layer, start + count - 1
                )
                output = tiered.attend(
                    (tiered_state,), layer, query, (1,), phase="decode"
                ).output
                if device_value.type == "cuda":
                    torch.cuda.synchronize(device_value)
                expected = dense_outputs[output_index].to(output.device)
                output_index += 1
                error = float(
                    (expected.float() - output.float()).abs().max().item()
                )
                max_error = max(max_error, error)
                if not torch.allclose(
                    output.float(), expected.float(), atol=2e-2, rtol=2e-2
                ):
                    raise AssertionError(
                        "tiered exact output differs from GPU-only dense: "
                        "max_error={}".format(error)
                    )
            tiered_latencies.append((time.perf_counter() - began) * 1000.0)
            start += count
        profile = tiered.profile_stats()
        required_nonzero = (
            "eviction_count",
            "prefetch_count",
            "d2h_kv_bytes",
            "h2d_kv_bytes",
        )
        missing = [name for name in required_nonzero if int(profile[name]) <= 0]
        if missing:
            raise AssertionError("Active Tier path did not exercise {}".format(missing))
        if profile["pending_tier_operations"]:
            raise AssertionError("Active Tier operations did not quiesce")
        result = {
            "hot_pages": hot_pages,
            "logical_pages": logical_pages,
            "page_size": page_size,
            "request_kv_larger_than_hot_cache": logical_pages > hot_pages,
            "max_abs_output_error": max_error,
            "dense_chunk_latency_ms": {
                "samples": dense_latencies,
                "mean": statistics.mean(dense_latencies),
                "p50": _percentile(dense_latencies, 0.50),
                "p90": _percentile(dense_latencies, 0.90),
            },
            "tiered_chunk_latency_ms": {
                "samples": tiered_latencies,
                "mean": statistics.mean(tiered_latencies),
                "p50": _percentile(tiered_latencies, 0.50),
                "p90": _percentile(tiered_latencies, 0.90),
            },
            "dense_gpu_peak_allocated_bytes": dense_peak_allocated,
            "dense_gpu_peak_reserved_bytes": dense_peak_reserved,
            "dense_gpu_baseline_allocated_bytes": dense_baseline_allocated,
            "dense_gpu_baseline_reserved_bytes": dense_baseline_reserved,
            "dense_gpu_peak_allocated_delta_bytes": (
                dense_peak_allocated - dense_baseline_allocated
                if dense_peak_allocated is not None
                else None
            ),
            "dense_gpu_peak_reserved_delta_bytes": (
                dense_peak_reserved - dense_baseline_reserved
                if dense_peak_reserved is not None
                else None
            ),
            "tiered_gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device_value))
                if device_value.type == "cuda"
                else None
            ),
            "tiered_gpu_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device_value))
                if device_value.type == "cuda"
                else None
            ),
            "tiered_gpu_baseline_allocated_bytes": tiered_baseline_allocated,
            "tiered_gpu_baseline_reserved_bytes": tiered_baseline_reserved,
            "profile": profile,
            "performance_class": "diagnostic_smoke_only",
        }
        if device_value.type == "cuda":
            result["tiered_gpu_peak_allocated_delta_bytes"] = (
                result["tiered_gpu_peak_allocated_bytes"]
                - tiered_baseline_allocated
            )
            result["tiered_gpu_peak_reserved_delta_bytes"] = (
                result["tiered_gpu_peak_reserved_bytes"]
                - tiered_baseline_reserved
            )
        return result
    finally:
        tiered.close()


class _SessionExecutor:
    def __init__(self, cache, device):
        self.kv_cache = cache
        self.device = torch.device(device)
        self.finish_calls = 0

    def begin(self, input_ids):
        return SimpleNamespace(input_ids=input_ids.to(self.device))

    def run_step(self, state):
        count = int(state.input_ids.shape[1])
        start = self.kv_cache.sequence_length()
        positions = torch.arange(
            start, start + count, device=self.device
        ).reshape(1, -1)
        for layer in range(2):
            key, value = _payload(
                self.device, torch.float16, layer, start, count
            )
            self.kv_cache.append_only(
                layer,
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
            )
            query = torch.ones(
                (1, 4, count, 8),
                dtype=torch.float16,
                device=self.device,
            ).mul_(float(layer + 1))
            self.kv_cache.attend(
                layer,
                query,
                kv_groups=2,
                position_ids=positions,
            )
        return state

    def finish(self, state):
        self.finish_calls += 1
        token = 100 + self.finish_calls
        state.topk_values = torch.tensor(
            [[[3.0, 2.0]]], device=self.device
        )
        state.topk_indices = torch.tensor(
            [[[token, token + 1]]], dtype=torch.long, device=self.device
        )
        return state


class _SessionModelRuntime:
    def run(self, executor, state):
        return executor.run_step(state)


def run_session_case(device):
    hot_pages, logical_pages, page_size = 4, 16, 16
    runtime = _runtime(
        device,
        tiered=True,
        hot_pages=hot_pages,
        logical_pages=logical_pages,
        page_size=page_size,
    )
    state = runtime.create_request(logical_pages * page_size)
    cache = RequestKVCacheV1(runtime, state)
    executor = _SessionExecutor(cache, device)
    session = GenerationSession(
        executor,
        _SessionModelRuntime(),
        SamplingConfig(top_k=1, max_new_tokens=4),
        eos_token_ids=(),
        input_device=device,
    )
    try:
        session.prefill(torch.arange(63, device=torch.device(device)))
        before = runtime.profile_stats()
        session.decode_one()
        session.decode_one()
        session.continue_prefill([70, 71])
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))
        after = runtime.profile_stats()
        if after["prefetch_count"] <= before["prefetch_count"]:
            raise AssertionError("GenerationSession did not prefetch evicted KV")
        if after["pending_tier_operations"]:
            raise AssertionError("GenerationSession left pending Tier work")
        session.reset()
        reset = runtime.profile_stats()
        if any(
            reset[name]
            for name in (
                "gpu_kv_used_pages",
                "cpu_kv_used_bytes",
                "kv_pool_allocated_pages",
                "pending_tier_operations",
            )
        ):
            raise AssertionError("GenerationSession reset did not reclaim Tier KV")
        return {
            "hot_pages": hot_pages,
            "logical_pages": logical_pages,
            "page_size": page_size,
            "request_kv_larger_than_hot_cache": True,
            "session_api_transparent": True,
            "prefetch_count_before_decode": before["prefetch_count"],
            "prefetch_count_after_decode": after["prefetch_count"],
            "h2d_kv_bytes": after["h2d_kv_bytes"],
            "d2h_kv_bytes": after["d2h_kv_bytes"],
            "reset_reclaimed_all_kv": True,
            "profile": after,
            "performance_class": "diagnostic_smoke_only",
        }
    finally:
        session.close()
        runtime.close()


def render_report(summary, cases):
    lines = [
        "# Active Tier Validation",
        "",
        "- Mode: `{}`".format(summary["mode"]),
        "- Status: `{}`".format(summary["status"]),
        "- Capability state: `{}`".format(summary["capability_state"]),
        "- Evidence class: `{}`".format(summary["evidence_class"]),
        "- Qualification admitted: `{}`".format(
            str(summary.get("qualification_admitted", False)).lower()
        ),
        "- Real-model performance claim: `false`",
        "- CUDA one-time initialization warmup: `{}`".format(
            str(summary.get("warmup_executed", False)).lower()
        ),
        "",
        "| Case | Status | Hot / Logical | Dense / Tiered peak allocated delta | Tiered mean chunk ms | Max error | Reason |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        metrics = case.get("metrics", {})
        lines.append(
            "| {} | `{}` | {} / {} | {} / {} | {} | {} | {} |".format(
                case["case_id"],
                case["status"],
                metrics.get("hot_pages", "-"),
                metrics.get("logical_pages", "-"),
                metrics.get("dense_gpu_peak_allocated_delta_bytes", "-"),
                metrics.get("tiered_gpu_peak_allocated_delta_bytes", "-"),
                metrics.get("tiered_chunk_latency_ms", {}).get("mean", "-"),
                metrics.get("max_abs_output_error", "-"),
                str(case.get("reason") or "-").replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "All timings and memory values are diagnostic in logic/cuda-smoke "
            "mode. Qualification mode requires scheduler-backed GPU allocation "
            "evidence and clean start/end process snapshots.",
            "",
            "This synthetic runner does not execute a real 70B model and cannot "
            "promote Tiered Dense KV beyond QUALIFICATION_READY. The remaining "
            "real-model A/B and full CUDA fault/long-stability matrix are reported "
            "as separate blockers.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("logic", "cuda-smoke", "qualification"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", default=str(ROOT / "reports" / "kv_active_tier")
    )
    args = parser.parse_args(argv)
    environment = capture_cuda_environment(ROOT, args.device)
    qualification_admitted = False
    if args.mode == "qualification":
        qualification_admitted, _, admission_reason = qualification_admission(
            environment, require_reservation=True
        )
        if not qualification_admitted:
            case_ids = [
                "ACTIVE-TIER-H{}-L{}-P{}".format(*configuration)
                for configuration in CASE_MATRIX
            ] + [SESSION_CASE_ID]
            cases = [
                {
                    "case_id": case_id,
                    "status": BLOCKED,
                    "evidence": "QUALIFICATION_READY",
                    "reason": admission_reason,
                    "metrics": {},
                }
                for case_id in case_ids
            ]
            blocked_status = (
                "BLOCKED_NO_RESERVATION"
                if "BLOCKED_NO_RESERVATION" in str(admission_reason)
                else BLOCKED
            )
            summary = {
                "schema_version": 1,
                "generated_at": utc_now(),
                "mode": args.mode,
                "status": blocked_status,
                "capability_state": "CUDA_SMOKE",
                "evidence_class": "PLAN_ONLY",
                "case_count": len(cases),
                "pass_count": 0,
                "blocked_count": len(cases),
                "qualification_admitted": False,
                "real_model_executed": False,
                "performance_qualified": False,
                "fault_matrix_complete": False,
                "warmup_executed": False,
            }
            environment["qualification_execution_authorized"] = False
            paths = write_report_bundle(
                args.output_dir,
                environment,
                cases,
                summary,
                render_report(summary, cases),
            )
            print(paths["report.md"])
            return 2
    environment["qualification_execution_authorized"] = qualification_admitted
    if args.mode in {"cuda-smoke", "qualification"} and not environment.get(
        "cuda_available"
    ):
        raise SystemExit("CUDA is required for {} mode".format(args.mode))
    device = "cpu" if args.mode == "logic" else args.device
    evidence = (
        "LOGIC_VALIDATED"
        if args.mode == "logic"
        else "SMOKE_ONLY"
        if args.mode == "cuda-smoke"
        else "EXCLUSIVE_CUDA_SYNTHETIC"
    )
    warmup_executed = False
    if args.mode in {"cuda-smoke", "qualification"}:
        # Keep one-time CUDA library/NVRTC allocation out of the configuration
        # deltas below.  This warmup is diagnostic and is reported explicitly.
        run_case(device, hot_pages=2, logical_pages=3, page_size=16)
        warmup_executed = True
    cases = []
    for hot_pages, logical_pages, page_size in CASE_MATRIX:
        case_id = "ACTIVE-TIER-H{}-L{}-P{}".format(
            hot_pages, logical_pages, page_size
        )
        try:
            metrics = run_case(
                device,
                hot_pages=hot_pages,
                logical_pages=logical_pages,
                page_size=page_size,
            )
        except BaseException as error:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "FAIL",
                    "evidence": evidence,
                    "reason": "{}: {}".format(type(error).__name__, error),
                    "metrics": {},
                }
            )
        else:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "PASS",
                    "evidence": evidence,
                    "reason": None,
                    "metrics": metrics,
                }
            )
    try:
        session_metrics = run_session_case(device)
    except BaseException as error:
        cases.append(
            {
                "case_id": SESSION_CASE_ID,
                "status": "FAIL",
                "evidence": evidence,
                "reason": "{}: {}".format(type(error).__name__, error),
                "metrics": {},
            }
        )
    else:
        cases.append(
            {
                "case_id": SESSION_CASE_ID,
                "status": "PASS",
                "evidence": evidence,
                "reason": None,
                "metrics": session_metrics,
            }
        )
    passed = sum(item["status"] == "PASS" for item in cases)
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "mode": args.mode,
        "status": "PASS" if passed == len(cases) else "FAIL",
        "capability_state": (
            "LOGIC_VALIDATED"
            if args.mode == "logic"
            else "CUDA_SMOKE"
            if args.mode == "cuda-smoke"
            else "QUALIFICATION_READY"
        ),
        "evidence_class": evidence,
        "case_count": len(cases),
        "pass_count": passed,
        "qualification_admitted": qualification_admitted,
        "real_model_executed": False,
        "performance_qualified": False,
        "fault_matrix_complete": False,
        "warmup_executed": warmup_executed,
    }
    end_environment = capture_cuda_environment(ROOT, args.device)
    environment["end_snapshot"] = {
        "captured_at": end_environment.get("captured_at"),
        "selected_physical_gpu": end_environment.get("selected_physical_gpu"),
        "selected_gpu_compute_processes": end_environment.get(
            "selected_gpu_compute_processes", []
        ),
        "selected_gpu_external_compute_processes": end_environment.get(
            "selected_gpu_external_compute_processes"
        ),
        "exclusive_snapshot": end_environment.get("exclusive_snapshot"),
        "reservation_evidence_present": end_environment.get(
            "reservation_evidence_present", False
        ),
        "reservation_evidence": end_environment.get("reservation_evidence"),
    }
    environment["external_activity_observed_at_sampled_boundaries"] = bool(
        environment.get("selected_gpu_external_compute_processes")
        or end_environment.get("selected_gpu_external_compute_processes")
    )
    if args.mode == "qualification":
        start_gpu = environment.get("selected_physical_gpu") or {}
        end_gpu = end_environment.get("selected_physical_gpu") or {}
        start_job = (environment.get("reservation_evidence") or {}).get("job_id")
        end_job = (end_environment.get("reservation_evidence") or {}).get("job_id")
        invalid_reason = None
        if environment["external_activity_observed_at_sampled_boundaries"]:
            invalid_reason = "INVALIDATED_EXTERNAL_GPU_ACTIVITY"
        elif start_gpu.get("uuid") != end_gpu.get("uuid"):
            invalid_reason = "INVALIDATED_GPU_IDENTITY_CHANGED"
        elif not end_environment.get("reservation_evidence_present", False):
            invalid_reason = "INVALIDATED_RESERVATION_EVIDENCE_LOST"
        elif start_job != end_job:
            invalid_reason = "INVALIDATED_RESERVATION_ID_CHANGED"
        if invalid_reason is not None:
            for case in cases:
                if case["status"] == "PASS":
                    case["status"] = "FAIL"
                    case["reason"] = invalid_reason
            summary["status"] = "FAIL"
            summary["pass_count"] = 0
            summary["capability_state"] = "CUDA_SMOKE"
    paths = write_report_bundle(
        args.output_dir,
        environment,
        cases,
        summary,
        render_report(summary, cases),
    )
    print(paths["report.md"])
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
