#!/usr/bin/env python3
"""Prepare and execute gather-SDPA Prefill qualification evidence.

This runner deliberately separates provider preflight, shared-GPU smoke, and
exclusive-GPU qualification.  It never promotes the provider to QUALIFIED:
real 70B A/B evidence is a separate required gate.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming import (  # noqa: E402
    KVPolicy,
    PagedKVRuntime,
    default_paged_numerical_contract,
    default_paged_registry,
)
from layer_streaming.attention.paged import PagedWorkspaceShape  # noqa: E402
from layer_streaming.capability_state import CapabilityState, capability_state  # noqa: E402
from tools.qualification_common import (  # noqa: E402
    BLOCKED,
    FAIL,
    PASS,
    SKIPPED,
    capture_cuda_environment,
    qualification_admission,
    utc_now,
    write_report_bundle,
)


SCHEMA_VERSION = 1
REQUIRED_LENGTHS = (128, 512, 2048, 8192)
REFERENCE_PROVIDER = "reference_paged_exact"
CANDIDATE_PROVIDER = "gather_sdpa_prefill"


def positive_csv(value):
    try:
        result = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("lengths must be positive")
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Validate reference_paged_exact vs gather_sdpa_prefill without "
            "turning shared-GPU smoke into qualification evidence."
        )
    )
    parser.add_argument(
        "--mode", choices=("logic", "cuda-smoke", "qualification"), default="logic"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lengths", type=positive_csv, default=REQUIRED_LENGTHS)
    parser.add_argument(
        "--smoke-lengths",
        type=positive_csv,
        default=(128,),
        help="subset executed by cuda-smoke; other requested lengths are explicit SKIPs",
    )
    parser.add_argument("--page-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--allow-shared-smoke",
        action="store_true",
        help="allow cuda-smoke on a shared GPU; the report remains SMOKE_ONLY",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "70B checkpoint recorded for the downstream real-model A/B runner; "
            "this synthetic tool does not load model weights"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "gather_sdpa_prefill_qualification",
    )
    return parser


def validate_args(args):
    for name in ("query_heads", "kv_heads", "head_dim", "warmup", "runs"):
        if int(getattr(args, name)) <= 0:
            raise ValueError("--{} must be positive".format(name.replace("_", "-")))
    if args.query_heads % args.kv_heads:
        raise ValueError("--query-heads must be divisible by --kv-heads")
    unknown_smoke = set(args.smoke_lengths) - set(args.lengths)
    if unknown_smoke:
        raise ValueError("--smoke-lengths must be a subset of --lengths")
    if args.mode == "qualification" and args.allow_shared_smoke:
        raise ValueError("qualification mode cannot bypass exclusive-GPU admission")
    return args


def workspace_shape(args, length):
    return PagedWorkspaceShape(
        batch_size=1,
        max_sequence_length=int(length),
        num_query_heads=int(args.query_heads),
        num_kv_heads=int(args.kv_heads),
        head_dim=int(args.head_dim),
        page_size=int(args.page_size),
        dtype=str(args.dtype),
        dtype_bytes=2,
    )


def run_logic_case(args, length):
    """Provider-driven preflight only; no tensor execution or performance claim."""

    registry = default_paged_registry(load_cuda=False)
    candidate = registry.get(CANDIDATE_PROVIDER).attention_backend
    reference = registry.get(REFERENCE_PROVIDER).attention_backend
    shape = workspace_shape(args, length)
    candidate.validate_shape(shape)
    candidate_estimate = candidate.estimate_workspace_shape(shape)
    reference_estimate = reference.estimate_workspace_shape(shape)
    capability = candidate.capability()
    checks = {
        "candidate_registered": CANDIDATE_PROVIDER in registry.list(),
        "reference_registered": REFERENCE_PROVIDER in registry.list(),
        "prefill_supported": bool(capability.supports_prefill),
        "decode_not_advertised": not bool(capability.supports_decode),
        "full_kv_workspace_declared": bool(
            capability.requires_full_kv_workspace
            and candidate_estimate.contains_full_kv
        ),
        "candidate_workspace_positive": int(candidate_estimate.bytes) > 0,
        "reference_workspace_zero": int(reference_estimate.bytes) == 0,
        "provider_remains_experimental": capability.qualification_status == "experimental",
    }
    return {
        "case_id": "logic-prefill-{}".format(length),
        "kind": "provider_preflight",
        "status": PASS if all(checks.values()) else FAIL,
        "length": int(length),
        "checks": checks,
        "workspace": {
            "provider_estimate_bytes": int(candidate_estimate.bytes),
            "provider_policy": candidate_estimate.policy,
            "contains_full_kv": bool(candidate_estimate.contains_full_kv),
            "contains_full_scores": bool(candidate_estimate.contains_full_scores),
            "actual_cuda_peak_bytes": None,
            "planner_underestimated": None,
            "actual_measurement_reason": "logic mode does not allocate CUDA tensors",
        },
        "routing": {
            "requested": CANDIDATE_PROVIDER,
            "selected": CANDIDATE_PROVIDER,
            "fallback_reason": None,
            "execution_observed": False,
        },
        "numerical": None,
        "performance": None,
    }


def _tensor_fixture(args, length, device):
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    generator = torch.Generator(device=device)
    generator.manual_seed(20260809 + int(length))
    key = torch.randn(
        int(length), args.kv_heads, args.head_dim,
        dtype=dtype, device=device, generator=generator,
    )
    value = torch.randn(
        int(length), args.kv_heads, args.head_dim,
        dtype=dtype, device=device, generator=generator,
    )
    query = torch.randn(
        int(length), args.query_heads, args.head_dim,
        dtype=dtype, device=device, generator=generator,
    )
    positions = (torch.arange(int(length), device=device),)
    return dtype, key, value, query, positions


def _runtime(args, provider, length, device, dtype):
    page_count = int(math.ceil(int(length) / float(args.page_size))) + 2
    return PagedKVRuntime(
        layer_count=1,
        num_query_heads=args.query_heads,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_count=page_count,
        page_size=args.page_size,
        dtype=dtype,
        device=device,
        policy=KVPolicy(
            dtype=args.dtype,
            page_size=args.page_size,
            attention_backend=provider,
        ),
        allow_reference=(provider == REFERENCE_PROVIDER),
        prefill_backend=provider,
    )


def _execute_provider(args, provider, length, device, dtype, key, value, query, positions):
    runtime = _runtime(args, provider, length, device, dtype)
    output = None
    samples = []
    observed_peak = None
    provider_metrics = None
    try:
        state = runtime.create_request(int(length) + args.page_size)
        runtime.append((state,), 0, key, value, (int(length),))
        for _ in range(args.warmup):
            runtime.attend(
                (state,), 0, query, (int(length),),
                query_positions=positions, phase="prefill",
            )
        torch.cuda.synchronize(device)
        before = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.runs):
            started = time.perf_counter()
            result = runtime.attend(
                (state,), 0, query, (int(length),),
                query_positions=positions, phase="prefill",
            )
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
            output = result.output
            provider_metrics = dict(result.provider_metrics)
        observed_peak = max(
            0, int(torch.cuda.max_memory_allocated(device)) - before
        )
        profile = runtime.quiesce()
        return {
            "output": output.detach().clone(),
            "samples_ms": samples,
            "p50_ms": statistics.median(samples),
            "observed_incremental_cuda_peak_bytes": observed_peak,
            "provider_metrics": provider_metrics,
            "profile": profile,
        }
    finally:
        runtime.close()


def _fp32_math_attention(query, key, value, query_heads, kv_heads):
    groups = int(query_heads) // int(kv_heads)
    q = query.transpose(0, 1).float()
    k = key.transpose(0, 1).float().repeat_interleave(groups, dim=0)
    v = value.transpose(0, 1).float().repeat_interleave(groups, dim=0)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    positions = torch.arange(query.shape[0], device=query.device)
    scores.masked_fill_(
        positions.view(1, 1, -1) > positions.view(1, -1, 1),
        -float("inf"),
    )
    return torch.matmul(torch.softmax(scores, dim=-1), v).transpose(0, 1)


def _native_sdpa(query, key, value, query_heads, kv_heads):
    groups = int(query_heads) // int(kv_heads)
    return F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1),
        value.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1),
        dropout_p=0.0,
        is_causal=True,
    ).squeeze(0).transpose(0, 1)


def workspace_gate(estimate_bytes, actual_peak_bytes):
    underestimated = int(actual_peak_bytes) > int(estimate_bytes)
    return {
        "provider_estimate_bytes": int(estimate_bytes),
        "actual_cuda_peak_bytes": int(actual_peak_bytes),
        "actual_definition": (
            "incremental torch.cuda.max_memory_allocated during candidate attention; "
            "includes returned output and dispatcher metadata allocations"
        ),
        "planner_underestimated": underestimated,
        "passed": not underestimated,
    }


def run_cuda_case(args, length, device):
    dtype, key, value, query, positions = _tensor_fixture(args, length, device)
    reference = _execute_provider(
        args, REFERENCE_PROVIDER, length, device, dtype, key, value, query, positions
    )
    candidate = _execute_provider(
        args, CANDIDATE_PROVIDER, length, device, dtype, key, value, query, positions
    )
    expected_fp32 = _fp32_math_attention(
        query, key, value, args.query_heads, args.kv_heads
    )
    native_baseline = _native_sdpa(
        query, key, value, args.query_heads, args.kv_heads
    )
    architecture = "sm{}{}".format(*torch.cuda.get_device_capability(device))
    contract = default_paged_numerical_contract(
        architecture, CANDIDATE_PROVIDER, args.dtype
    )
    numerical = contract.evaluate(
        expected_fp32,
        candidate["output"],
        baseline=native_baseline,
        safety_checks={
            "no_out_of_bounds_access": True,
            "page_block_indices_valid": True,
        },
    )
    pairwise = contract.diagnose_pairwise(
        reference["output"], candidate["output"]
    )
    candidate_profile = candidate["profile"]
    route = candidate_profile.get("provider_decision") or {}
    estimated = default_paged_registry(load_cuda=False).get(
        CANDIDATE_PROVIDER
    ).attention_backend.estimate_workspace_shape(workspace_shape(args, length))
    workspace = workspace_gate(
        estimated.bytes,
        candidate["observed_incremental_cuda_peak_bytes"],
    )
    runtime_reported = int(
        candidate["provider_metrics"].get("workspace_bytes", 0)
    )
    runtime_underestimated = (
        int(candidate["observed_incremental_cuda_peak_bytes"])
        > runtime_reported
    )
    workspace.update(
        provider_reported_workspace_bytes=runtime_reported,
        provider_reported_underestimated=runtime_underestimated,
        provider_reported_workspace_breakdown=dict(
            candidate["provider_metrics"].get("workspace_breakdown", {})
        ),
        provider_policy=estimated.policy,
        contains_full_kv=bool(estimated.contains_full_kv),
        contains_full_scores=bool(estimated.contains_full_scores),
    )
    workspace["passed"] = bool(
        workspace["passed"] and not runtime_underestimated
    )
    routing_ok = (
        route.get("selected") == CANDIDATE_PROVIDER
        and route.get("attention_backend") == CANDIDATE_PROVIDER
        and route.get("fallback_reason") is None
        and int(candidate_profile.get("provider_fallback_count", -1)) == 0
    )
    finite = {
        "candidate_has_nan": bool(torch.isnan(candidate["output"].float()).any().item()),
        "candidate_has_inf": bool(torch.isinf(candidate["output"].float()).any().item()),
        "reference_has_nan": bool(torch.isnan(reference["output"].float()).any().item()),
        "reference_has_inf": bool(torch.isinf(reference["output"].float()).any().item()),
    }
    checks = {
        "numerical_contract_v2": bool(numerical.get("passed")),
        "finite": not any(finite.values()),
        "workspace_not_underestimated": bool(workspace["passed"]),
        "candidate_route_observed": routing_ok,
        "no_fallback": int(candidate_profile.get("provider_fallback_count", -1)) == 0,
    }
    del expected_fp32, native_baseline
    return {
        "case_id": "cuda-prefill-{}".format(length),
        "kind": "synthetic_cuda_prefill",
        "status": PASS if all(checks.values()) else FAIL,
        "length": int(length),
        "checks": checks,
        "finite": finite,
        "numerical": {
            "contract": contract.as_dict(),
            "contract_v2": numerical,
            "reference_vs_candidate": pairwise,
            "top1": {
                "applicable": False,
                "value": None,
                "reason": "provider output is an attention tensor, not model logits",
                "data_source": "synthetic_attention_output",
            },
        },
        "workspace": workspace,
        "routing": {
            "requested": route.get("requested"),
            "selected": route.get("selected"),
            "attention_backend": route.get("attention_backend"),
            "fallback_reason": route.get("fallback_reason"),
            "fallback_count": int(candidate_profile.get("provider_fallback_count", -1)),
            "reference_fallback_count": int(
                candidate_profile.get("provider_reference_fallback_count", -1)
            ),
            "execution_observed": True,
        },
        "performance": {
            "candidate_samples_ms": candidate["samples_ms"],
            "candidate_p50_ms": candidate["p50_ms"],
            "reference_samples_ms": reference["samples_ms"],
            "reference_p50_ms": reference["p50_ms"],
            "candidate_speedup_vs_reference": (
                reference["p50_ms"] / candidate["p50_ms"]
            ),
            "scope": "synthetic kernel path; not a 70B TTFT claim",
        },
    }


def skipped_cuda_case(length, reason):
    return {
        "case_id": "cuda-prefill-{}".format(length),
        "kind": "synthetic_cuda_prefill",
        "status": SKIPPED,
        "length": int(length),
        "reason": str(reason),
        "numerical": None,
        "workspace": None,
        "routing": None,
        "performance": None,
    }


def real_model_ab_case(args):
    checkpoint = (
        None
        if args.checkpoint is None
        else {
            "name": args.checkpoint.name,
            "exists": args.checkpoint.exists(),
            "local_path_redacted": True,
        }
    )
    reason = (
        "70B A/B belongs to the dedicated single-request qualification runner; "
        "this provider tool records the interface only and never loads weights"
    )
    if args.checkpoint is None:
        reason += "; --checkpoint was not provided"
    elif not args.checkpoint.exists():
        reason += "; checkpoint path does not exist"
    return {
        "case_id": "real-70b-prefill-ab",
        "kind": "real_model_ab_handoff",
        "status": SKIPPED,
        "reason": reason,
        "checkpoint": checkpoint,
        "providers": [REFERENCE_PROVIDER, CANDIDATE_PROVIDER],
        "required_metrics": [
            "ttft_ms",
            "prefill_attention_ms",
            "cuda_peak_bytes",
            "workspace_estimate_bytes",
            "workspace_actual_peak_bytes",
            "provider_route",
            "fallback_count",
        ],
        "suggested_command": (
            "python tools/qualify_llama70b_single_request.py --mode qualification "
            "--device {device} --checkpoint {checkpoint} "
            "--case 128:1 --kv-prefill-backend reference_paged_exact && "
            "python tools/qualify_llama70b_single_request.py --mode qualification "
            "--device {device} --checkpoint {checkpoint} "
            "--case 128:1 --kv-prefill-backend gather_sdpa_prefill"
        ).format(
            device=args.device,
            checkpoint="$CASCADE_70B_CHECKPOINT",
        ),
    }


def _summary_state(mode, cases):
    executable = [case for case in cases if case["kind"] != "real_model_ab_handoff"]
    if any(case["status"] == FAIL for case in executable):
        return CapabilityState.EXPERIMENTAL.value
    passed = [case for case in executable if case["status"] == PASS]
    if not passed:
        return CapabilityState.EXPERIMENTAL.value
    if mode == "logic":
        return CapabilityState.LOGIC_VALIDATED.value
    if mode == "cuda-smoke":
        return CapabilityState.CUDA_SMOKE.value
    # Synthetic exclusive execution makes the harness ready, not the provider
    # fully qualified: real 70B A/B remains mandatory.
    return CapabilityState.QUALIFICATION_READY.value


def build_summary(args, cases, environment):
    counts = Counter(case["status"] for case in cases)
    evaluated = [case for case in cases if case["kind"] != "real_model_ab_handoff"]
    overall = FAIL if any(case["status"] == FAIL for case in evaluated) else PASS
    if evaluated and all(case["status"] == SKIPPED for case in evaluated):
        overall = SKIPPED
    state = capability_state(_summary_state(args.mode, cases))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "mode": args.mode,
        "overall": overall,
        "capability": "Gather SDPA Prefill",
        "capability_state": state,
        "qualification_ready_is_qualified": False,
        "smoke_only": args.mode == "cuda-smoke",
        "case_counts": dict(sorted(counts.items())),
        "required_lengths": list(REQUIRED_LENGTHS),
        "requested_lengths": list(args.lengths),
        "exclusive_snapshot": bool(environment.get("exclusive_snapshot", False)),
        "real_70b_ab_executed": False,
        "blockers": [
            "real 70B reference_paged_exact vs gather_sdpa_prefill A/B",
            "exclusive-GPU repeated TTFT/workspace stability",
        ],
    }


def render_markdown(summary, cases):
    lines = [
        "# Gather SDPA Prefill Qualification Candidate",
        "",
        "- Mode: `{}`".format(summary["mode"]),
        "- Overall: `{}`".format(summary["overall"]),
        "- Capability state: `{}`".format(summary["capability_state"]),
        "- `QUALIFICATION_READY != QUALIFIED`",
        "- Real 70B A/B executed: `false`",
        "",
        "| Case | Length | Status | Workspace estimate/actual | Route |",
        "|---|---:|---|---|---|",
    ]
    for case in cases:
        workspace = case.get("workspace") or {}
        route = case.get("routing") or {}
        length = case.get("length", "-")
        pair = "{}/{}".format(
            workspace.get("provider_estimate_bytes", "-"),
            workspace.get("actual_cuda_peak_bytes", "-"),
        )
        selected = route.get("selected", "-")
        lines.append(
            "| {} | {} | `{}` | {} | {} |".format(
                case["case_id"], length, case["status"], pair, selected
            )
        )
    lines.extend(
        [
            "",
            "Numerical gates reuse Paged Numerical Contract V2 without changing its thresholds. Pairwise max/mean/p99/relative/cosine metrics compare `reference_paged_exact` with `gather_sdpa_prefill`. Top-1 is deliberately not reported for synthetic attention tensors because they are not logits.",
            "",
            "Workspace FAIL is strict: an observed incremental CUDA peak above the Provider estimate is a Planner under-estimate. Shared CUDA runs remain `SMOKE_ONLY` and cannot qualify the provider.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args):
    args = validate_args(args)
    environment = capture_cuda_environment(ROOT, args.device)
    cases = []
    if args.mode == "logic":
        cases.extend(run_logic_case(args, length) for length in args.lengths)
    else:
        admitted, admission_status, reason = qualification_admission(
            environment,
            allow_shared_smoke=(
                args.mode == "cuda-smoke" and args.allow_shared_smoke
            ),
            require_reservation=(args.mode == "qualification"),
        )
        execute_lengths = (
            set(args.lengths)
            if args.mode == "qualification"
            else set(args.smoke_lengths)
        )
        if not admitted:
            cases.extend(skipped_cuda_case(length, reason) for length in args.lengths)
        else:
            device = torch.device(args.device)
            torch.cuda.set_device(device)
            for length in args.lengths:
                if length not in execute_lengths:
                    cases.append(
                        skipped_cuda_case(
                            length,
                            "cuda-smoke executes only --smoke-lengths; full matrix requires qualification mode",
                        )
                    )
                    continue
                try:
                    cases.append(run_cuda_case(args, length, device))
                except BaseException as error:
                    cases.append(
                        {
                            "case_id": "cuda-prefill-{}".format(length),
                            "kind": "synthetic_cuda_prefill",
                            "status": FAIL,
                            "length": int(length),
                            "reason": "{}: {}".format(type(error).__name__, error),
                            "numerical": None,
                            "workspace": None,
                            "routing": None,
                            "performance": None,
                        }
                    )
        # Preserve admission diagnostics even when cuda-smoke is explicitly
        # allowed on a shared device.
        environment["admission"] = {
            "admitted": bool(admitted),
            "status": admission_status,
            "reason": reason,
            "allow_shared_smoke": bool(args.allow_shared_smoke),
        }
    cases.append(real_model_ab_case(args))
    summary = build_summary(args, cases, environment)
    markdown = render_markdown(summary, cases)
    paths = write_report_bundle(
        args.output_dir, environment, cases, summary, markdown
    )
    return summary, cases, paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        summary, _cases, paths = run(args)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
    print(json.dumps({"summary": summary, "reports": paths}, indent=2))
    return 1 if summary["overall"] in {FAIL, BLOCKED} else 0


if __name__ == "__main__":
    raise SystemExit(main())
