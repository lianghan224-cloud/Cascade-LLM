#!/usr/bin/env python3
"""RGKV device-driven GPU-hit Decode gate.

Qualification requires an exclusive scheduler-backed CUDA allocation.  Logic
mode exercises the 1000-launch lifecycle/metrics harness with CPU tensors and
cannot qualify CUDA behavior.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from layer_streaming.attention.paged import (
    DevicePagedAttentionInput,
    PagedAttentionOutput,
)
from layer_streaming.kv.runtime import PagedKVRuntime
from layer_streaming.kv_policy import KVPolicy
from tools.qualification_common import (
    BLOCKED,
    BLOCKED_NOT_EXCLUSIVE,
    FAIL,
    PASS,
    SKIPPED,
    GPUActivityMonitor,
    capture_cuda_environment,
    qualification_admission,
    write_report_bundle,
)


SCHEMA_VERSION = 1
NOT_QUALIFIED = "NOT_QUALIFIED"
SMOKE_ONLY = "SMOKE_ONLY"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _cpu_factory(shape, **kwargs):
    return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


def _runtime(device, *, attention_backend, tensor_factory=None):
    page_size = 16
    layers, kv_heads, head_dim = 1, 1, 32
    logical_page_bytes = (
        2 * layers * kv_heads * page_size * head_dim
        * torch.empty((), dtype=torch.float16).element_size()
    )
    return PagedKVRuntime(
        layer_count=layers,
        num_query_heads=2,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        page_count=8,
        page_size=page_size,
        dtype=torch.float16,
        device=device,
        policy=KVPolicy(
            accuracy="sparse",
            storage="gpu_cpu",
            dtype="fp16",
            selection="rgkv",
            rgkv_scorer="torch_tensorized",
            page_budget=2,
            recent_window=page_size,
            attention_backend=attention_backend,
            page_size=page_size,
            gpu_hot_budget_bytes=4 * logical_page_bytes,
            cpu_budget_bytes=8 * logical_page_bytes,
            gpu_high_watermark_bytes=4 * logical_page_bytes,
            gpu_low_watermark_bytes=3 * logical_page_bytes,
        ),
        allow_reference=(attention_backend == "reference_paged_exact"),
        tier_tensor_factory=tensor_factory,
    )


def _prepare(runtime):
    state = runtime.create_request(32)
    key = torch.randn(
        32, runtime.num_kv_heads, runtime.head_dim,
        dtype=runtime.dtype, device=runtime.device,
    )
    runtime.append((state,), 0, key, key, (32,))
    query = torch.randn(
        1, runtime.num_query_heads, runtime.head_dim,
        dtype=runtime.dtype, device=runtime.device,
    )
    return state, query


def _logic_case(iterations):
    runtime = _runtime(
        "cpu",
        attention_backend="reference_paged_exact",
        tensor_factory=_cpu_factory,
    )
    try:
        state, query = _prepare(runtime)
        batch = runtime.prepare_batch((state,), (1,), 0)
        selected = runtime.selection.select((state,), 0, query, batch)
        key_pool, value_pool = runtime.store.layer_view(0)
        request = DevicePagedAttentionInput(
            query=query,
            key_pool_view=key_pool,
            value_pool_view=value_pool,
            batch_view=batch,
            page_size=runtime.page_size,
            num_query_heads=runtime.num_query_heads,
            num_kv_heads=runtime.num_kv_heads,
            head_dim=runtime.head_dim,
            softmax_scale=runtime.softmax_scale,
            causal=True,
            kv_dtype="fp16",
            output_dtype="fp16",
            selected_pages=selected,
        )

        def fake_device(current, *, phase):
            if phase != "decode":
                raise AssertionError("logic gate requires Decode")
            return PagedAttentionOutput(
                current.query.clone(), None, {"workspace_bytes": 0}
            )

        runtime.dispatcher.execute_device = fake_device
        started = time.perf_counter()
        for _ in range(iterations):
            guard = runtime.active_tier.begin_hot_cache_execution(
                request_id=state.request_id
            )
            runtime.execution._attend_active_tier_device_hit(
                state, 0, request, guard
            )
        runtime.active_tier.quiesce(request_id=state.request_id)
        stats = runtime.profile_stats()
        failures = []
        for name in (
            "selected_metadata_d2h",
            "host_scalar_readbacks",
            "explicit_sync_count",
            "python_attention_wave_count",
            "device_view_fallback_count",
            "active_hot_cache_execution_guards",
        ):
            if int(stats.get(name, -1)) != 0:
                failures.append("{}={}".format(name, stats.get(name)))
        if stats["total_ref_count"] != stats["logical_owner_count"]:
            failures.append("ref_count != logical_owner_count")
        if stats["gpu_hit_device_attention_calls"] != iterations:
            failures.append("device attention call count mismatch")
        return {
            "id": "RGKV-DEVICE-1000-LOGIC",
            "status": PASS if not failures else FAIL,
            "qualification": NOT_QUALIFIED,
            "iterations": iterations,
            "duration_seconds": time.perf_counter() - started,
            "metrics": {
                name: stats[name] for name in (
                    "gpu_hit_device_attention_calls",
                    "selected_metadata_d2h",
                    "host_scalar_readbacks",
                    "explicit_sync_count",
                    "python_attention_wave_count",
                    "device_view_fallback_count",
                    "active_hot_cache_execution_guards",
                    "total_ref_count",
                    "logical_owner_count",
                )
            },
            "failures": failures,
            "evidence_scope": "CPU synthetic orchestration only",
        }
    finally:
        runtime.close()


def _cuda_case(device, iterations):
    runtime = _runtime(device, attention_backend="generic_cuda")
    try:
        state, query = _prepare(runtime)
        started = time.perf_counter()
        output_hash = None
        for _ in range(iterations):
            result = runtime.attend(
                (state,), 0, query, (1,), phase="decode"
            )
            # No per-iteration scalar readback. Hashing is intentionally done
            # once at this final safe reporting boundary.
            output_hash = result.output.detach()
        runtime.quiesce()
        digest = __import__("hashlib").sha256(
            output_hash.cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        stats = runtime.profile_stats()
        failures = []
        for name in (
            "selected_metadata_d2h",
            "host_scalar_readbacks",
            "explicit_sync_count",
            "python_attention_wave_count",
            "device_view_fallback_count",
            "active_hot_cache_execution_guards",
        ):
            if int(stats.get(name, -1)) != 0:
                failures.append("{}={}".format(name, stats.get(name)))
        return {
            "id": "RGKV-DEVICE-1000-CUDA",
            "status": PASS if not failures else FAIL,
            "qualification": NOT_QUALIFIED,
            "iterations": iterations,
            "duration_seconds": time.perf_counter() - started,
            "output_hash": digest,
            "metrics": {name: stats[name] for name in (
                "gpu_hit_device_attention_calls",
                "selected_metadata_d2h",
                "host_scalar_readbacks",
                "explicit_sync_count",
                "python_attention_wave_count",
                "device_view_fallback_count",
                "active_hot_cache_execution_guards",
            )},
            "failures": failures,
        }
    finally:
        runtime.close()


def _markdown(summary, cases):
    lines = [
        "# RGKV Device-Driven Attention Gate",
        "",
        "- Mode: `{}`".format(summary["mode"]),
        "- Status: `{}`".format(summary["status"]),
        "- Qualification: `{}`".format(summary["qualification"]),
        "- Iterations: `{}`".format(summary["iterations"]),
        "",
        "| Case | Status | Qualification | Reason |",
        "|---|---|---|---|",
    ]
    for case in cases:
        lines.append("| {} | `{}` | `{}` | {} |".format(
            case["id"], case["status"], case["qualification"],
            case.get("reason") or "; ".join(case.get("failures", ())) or "none",
        ))
    lines.extend((
        "",
        "> CPU logic is not CUDA qualification. Shared GPU execution is SMOKE_ONLY;",
        "> formal qualification requires exclusive scheduler-backed allocation.",
        "",
    ))
    return "\n".join(lines)


def run(args):
    if args.mode == "logic":
        environment = {
            "captured_at": utc_now(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_executed": False,
        }
        cases = [_logic_case(args.iterations)]
        qualification = NOT_QUALIFIED
    else:
        environment = capture_cuda_environment(ROOT, args.device)
        admitted, status, reason = qualification_admission(
            environment,
            allow_shared_smoke=(args.mode == "cuda-smoke" and args.allow_shared_smoke),
            require_reservation=(args.mode == "qualification"),
        )
        if not admitted:
            cases = [{
                "id": "RGKV-DEVICE-1000-CUDA",
                "status": status,
                "qualification": NOT_QUALIFIED,
                "reason": reason,
            }]
            qualification = NOT_QUALIFIED
        else:
            selected = environment["selected_physical_gpu"]
            monitor = GPUActivityMonitor(selected["uuid"]).start()
            try:
                case = _cuda_case(args.device, args.iterations)
            finally:
                activity = monitor.stop()
            case["gpu_activity"] = activity
            if activity["external_activity_observed"]:
                case["status"] = "INVALIDATED_EXTERNAL_GPU_ACTIVITY"
                case["qualification"] = NOT_QUALIFIED
            elif args.mode == "cuda-smoke":
                case["qualification"] = SMOKE_ONLY
            else:
                case["qualification"] = "QUALIFICATION_EVIDENCE"
            cases = [case]
            qualification = case["qualification"]
    overall = cases[0]["status"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "mode": args.mode,
        "status": overall,
        "qualification": qualification,
        "iterations": args.iterations,
        "device_tier_miss_batch_started": False,
    }
    paths = write_report_bundle(
        args.output_dir, environment, cases, summary, _markdown(summary, cases)
    )
    return summary, cases, paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("logic", "cuda-smoke", "qualification"),
        default="logic",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--allow-shared-smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "rgkv_device_attention_gate_20260812"),
    )
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def main(argv=None):
    summary, _, paths = run(parse_args(argv))
    print(json.dumps({"summary": summary, "artifacts": paths}, indent=2))
    return 0 if summary["status"] in {PASS, SKIPPED, BLOCKED, BLOCKED_NOT_EXCLUSIVE} else 1


if __name__ == "__main__":
    raise SystemExit(main())
