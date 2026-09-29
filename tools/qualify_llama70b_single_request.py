#!/usr/bin/env python3
"""Unified 70B dense-KV single-request qualification runner.

The runner deliberately wraps ``tools/run_llama31.py`` instead of owning a
second inference implementation.  ``plan`` is side-effect free with respect
to the checkpoint and CUDA.  ``qualification`` performs a point-in-time
exclusive-GPU admission check and refuses to start when another compute
process is present.  ``--allow-shared-smoke`` changes the evidence class to
``SMOKE_ONLY``; it never turns shared-GPU data into qualification evidence.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qualification_common import (  # noqa: E402
    BLOCKED_NOT_EXCLUSIVE,
    GPUActivityMonitor,
    capture_cuda_environment,
    qualification_admission,
    utc_now,
    write_report_bundle,
)


SCHEMA_VERSION = 1
PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED_WITH_REASON"
SMOKE_ONLY = "SMOKE_ONLY"
PLAN_ONLY = "PLAN_ONLY"

DEFAULT_MATRIX = (
    (1, 1),
    (128, 32),
    (512, 32),
    (2048, 128),
    (8192, 32),
    (16384, 32),
    (128, 1000),
)
REQUIRED_PROMPT_LENGTHS = (1, 128, 512, 2048, 8192, 16384)
REQUIRED_DECODE_LENGTHS = (1, 32, 128, 1000)


def parse_case(value):
    try:
        prompt, decode = (int(item) for item in value.split(":", 1))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("case must be PROMPT_TOKENS:DECODE_TOKENS") from error
    if prompt <= 0 or decode <= 0:
        raise argparse.ArgumentTypeError("case lengths must be positive")
    return prompt, decode


def parse_byte_size(value):
    text = str(value).strip().lower()
    multiplier = 1
    for suffix, amount in (("gib", 1024 ** 3), ("mib", 1024 ** 2), ("b", 1)):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            multiplier = amount
            break
    try:
        result = int(float(text) * multiplier)
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid byte size") from error
    if result < 0:
        raise argparse.ArgumentTypeError("byte size cannot be negative")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan", "smoke", "qualification"), default="plan",
        help="plan never loads the checkpoint; smoke is never qualification evidence",
    )
    parser.add_argument("--dry-run", action="store_true", help="alias for --mode plan")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-shared-smoke", action="store_true",
        help="run despite external GPU processes and label every result SMOKE_ONLY",
    )
    parser.add_argument(
        "--require-uncontended-benchmark",
        action="store_true",
        help=(
            "require no external process before, during, or after each case; "
            "this is still not exclusive qualification evidence"
        ),
    )
    parser.add_argument(
        "--case", action="append", type=parse_case, dest="cases",
        help="override the default matrix; repeat as PROMPT_TOKENS:DECODE_TOKENS",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        help="repetitions per case; defaults to 5 for qualification, 1 otherwise",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "llama70b_single_request_qualification")
    parser.add_argument(
        "--raw-dir", type=Path,
        default=Path("/tmp/cascade_llama70b_single_request_qualification_raw"),
        help="external location for large child RunReports/stdout/stderr; keep raw traces out of Git",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--weight-store", choices=("full_pinned", "pinned_staging"), default="pinned_staging")
    parser.add_argument("--granularity", choices=("matrix", "matrix_group", "layer"), default="matrix")
    parser.add_argument("--slots", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--prefetch-depth", type=int, choices=tuple(range(0, 9)), default=2)
    parser.add_argument("--embedding-placement", choices=("resident", "streamed"), default="resident")
    parser.add_argument("--lm-head-placement", choices=("resident", "streamed"), default="streamed")
    parser.add_argument("--kv-page-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--kv-attention-backend", default="generic_cuda")
    parser.add_argument("--kv-prefill-backend", choices=("reference_paged_exact", "gather_sdpa_prefill"), default="gather_sdpa_prefill")
    parser.add_argument("--allow-kv-reference", action="store_true")
    parser.add_argument("--cuda-safety-margin-mib", type=int, default=512)
    parser.add_argument("--ignore-memlock-limit", action="store_true")
    parser.add_argument(
        "--gpu-resident-weight-budget",
        type=parse_byte_size,
        default=0,
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        args.mode = "plan"
    if args.repeat is None:
        args.repeat = 5 if args.mode == "qualification" else 1
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.mode != "plan" and args.checkpoint is None:
        parser.error("--checkpoint is required outside plan mode")
    if args.mode != "plan" and not args.checkpoint.is_dir():
        parser.error("--checkpoint must be an existing local directory")
    if args.mode == "smoke":
        args.allow_shared_smoke = True
    return args


def planned_cases(matrix=None, repeat=1):
    result = []
    for prompt_tokens, decode_tokens in tuple(matrix or DEFAULT_MATRIX):
        for repetition in range(1, repeat + 1):
            result.append(
                {
                    "case_id": "p{}_d{}_r{:02d}".format(prompt_tokens, decode_tokens, repetition),
                    "prompt_tokens": int(prompt_tokens),
                    "decode_tokens": int(decode_tokens),
                    "repetition": repetition,
                }
            )
    return result


def _metric(value, source, *, note=None):
    result = {"available": value is not None, "value": value, "source": source}
    if note:
        result["note"] = note
    return result


def _nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def percentile(values, percent):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(percent) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def normalize_run_report(report):
    """Normalize RunReport v2 without inventing unavailable child state."""

    timings = report.get("timings", {})
    throughput = report.get("throughput", {})
    pipeline = report.get("pipeline", {})
    kv = pipeline.get("kv", {})
    memory = report.get("memory", {})
    latencies = timings.get("decode_token_latencies_ms") or []
    per_forward = _nested(
        pipeline, "copy_compute_timeline", "per_forward"
    ) or []
    prefill_forward = next(
        (item for item in per_forward if item.get("phase") == "prefill"),
        {},
    )
    prefill_summary = prefill_forward.get("summary") or {}
    placement = report.get("transformer_placement") or {}
    generation = report.get("generation") or {}
    generated_ids = generation.get("generated_token_ids") or []
    output_hash = hashlib.sha256(
        json.dumps(generated_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    h2d_bytes = pipeline.get("h2d_bytes")
    h2d_ms = timings.get("h2d_time_ms")
    bandwidth = None
    if isinstance(h2d_bytes, (int, float)) and isinstance(h2d_ms, (int, float)) and h2d_ms > 0:
        bandwidth = float(h2d_bytes) / (float(h2d_ms) / 1000.0)
    missing_source = "unavailable_in_subprocess_RunReport_v2"
    return {
        "performance": {
            "model_load_seconds": _metric(timings.get("load_time_seconds"), "RunReport.timings.load_time_seconds"),
            "ttft_ms": _metric(timings.get("time_to_first_token_ms"), "RunReport.timings.time_to_first_token_ms"),
            "prefill_ms": _metric(
                timings.get("time_to_first_token_ms"),
                "TTFT proxy from RunReport; isolated prefill wall time unavailable",
            ),
            "prefill_attention_ms": _metric(kv.get("prefill_attention_ms"), "RunReport.pipeline.kv.prefill_attention_ms"),
            "decode_p50_ms": _metric(percentile(latencies, 50), "derived from RunReport.timings.decode_token_latencies_ms"),
            "decode_p90_ms": _metric(percentile(latencies, 90), "derived from RunReport.timings.decode_token_latencies_ms"),
            "tpot_mean_ms": _metric(
                (sum(float(item) for item in latencies) / len(latencies))
                if latencies else None,
                "mean RunReport.timings.decode_token_latencies_ms; unavailable for Decode=1",
            ),
            "prefill_transformer_ms": _metric(
                prefill_summary.get("gpu_timeline_span_ms"),
                "RunReport.pipeline.copy_compute_timeline prefill GPU span",
            ),
            "gpu_idle_or_host_overhead_ms": _metric(
                timings.get("gpu_idle_or_host_overhead_ms"),
                "RunReport.timings.gpu_idle_or_host_overhead_ms",
            ),
            "token_per_second": _metric(throughput.get("tokens_per_second"), "RunReport.throughput.tokens_per_second"),
            "decode_token_per_second": _metric(throughput.get("decode_tokens_per_second"), "RunReport.throughput.decode_tokens_per_second"),
        },
        "weight_streaming": {
            "h2d_bytes": _metric(h2d_bytes, "RunReport.pipeline.h2d_bytes"),
            "h2d_exposed_wait_ms": _metric(timings.get("unoverlapped_h2d_ms"), "RunReport.timings.unoverlapped_h2d_ms"),
            "copy_compute_overlap_ms": _metric(timings.get("h2d_compute_overlap_ms"), "RunReport.timings.h2d_compute_overlap_ms"),
            "slot_reuse": _metric(pipeline.get("transfer_slot_reuse_counts"), "RunReport.pipeline.transfer_slot_reuse_counts"),
            "effective_pcie_bandwidth_bytes_per_second": _metric(bandwidth, "derived h2d_bytes / h2d_time_ms"),
            "h2d_time_ms": _metric(
                h2d_ms, "RunReport.timings.h2d_time_ms"
            ),
            "h2d_bytes_per_generated_token": _metric(
                float(h2d_bytes) / max(1, int(throughput.get("generated_tokens") or 0))
                if isinstance(h2d_bytes, (int, float)) else None,
                "derived total h2d_bytes / generated_tokens",
            ),
            "resident_transformer_weight_bytes": _metric(
                placement.get("resident_weight_bytes"),
                "RunReport.transformer_placement.resident_weight_bytes",
            ),
            "resident_weight_arena_bytes": _metric(
                placement.get("resident_arena_bytes"),
                "RunReport.transformer_placement.resident_arena_bytes",
            ),
        },
        "kv": {
            "pages_total": _metric(kv.get("kv_pool_total_pages"), "RunReport.pipeline.kv.kv_pool_total_pages"),
            "pages_peak": _metric(kv.get("kv_pool_peak_pages"), "RunReport.pipeline.kv.kv_pool_peak_pages"),
            "ref_total": _metric(None, missing_source),
            "owner_total": _metric(None, missing_source),
            "pin_total": _metric(None, missing_source),
            "inflight_compute": _metric(None, missing_source),
            "inflight_io": _metric(None, missing_source),
            "kv_bytes": _metric(memory.get("kv_cache_bytes"), "RunReport.memory.kv_cache_bytes"),
            "workspace_estimate_bytes": _metric(memory.get("kv_attention_workspace_bytes"), "RunReport.memory.kv_attention_workspace_bytes"),
            "workspace_actual_bytes": _metric(kv.get("kv_workspace_peak_bytes"), "RunReport.pipeline.kv.kv_workspace_peak_bytes"),
            "provider_route": _metric(kv.get("provider_routing_summary"), "RunReport.pipeline.kv.provider_routing_summary"),
            "fallback_count": _metric(kv.get("provider_fallback_count"), "RunReport.pipeline.kv.provider_fallback_count"),
            "reference_fallback_count": _metric(kv.get("provider_reference_fallback_count"), "RunReport.pipeline.kv.provider_reference_fallback_count"),
        },
        "memory": {
            "cuda_allocated_peak_bytes": _metric(memory.get("gpu_peak_memory_bytes"), "RunReport.memory.gpu_peak_memory_bytes"),
            "cuda_reserved_peak_bytes": _metric(None, missing_source),
            "pinned_ram_bytes": _metric(memory.get("pinned_bytes"), "RunReport.memory.pinned_bytes"),
            "cpu_rss_peak_bytes": _metric(None, missing_source),
            "cpu_resident_arena_bytes": _metric(memory.get("cpu_resident_bytes"), "RunReport.memory.cpu_resident_bytes", note="arena size, not process RSS"),
        },
        "resource_snapshot": {
            "available": False,
            "source": "KVResourceSnapshot requires in-process KV runtime; child RunReport v2 normalized above",
            "api": "layer_streaming.kv.resource_audit.KVResourceSnapshot",
        },
        "output": {
            "sha256": output_hash,
            "token_count": len(generated_ids),
        },
        "fallback": {
            "backend_fallbacks": list(pipeline.get("fallback_backends") or []),
            "kv_provider_fallback_count": int(
                kv.get("provider_fallback_count") or 0
            ),
            "kv_reference_fallback_count": int(
                kv.get("provider_reference_fallback_count") or 0
            ),
        },
    }


def _build_prompt(checkpoint, target_tokens):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if target_tokens == 1:
        candidates = ("", " ")
    else:
        candidates = (
            " hello" * (target_tokens - 1),
            "x " * (target_tokens - 1),
            "a " * (target_tokens - 1),
        )
    for prompt in candidates:
        encoded = tokenizer(prompt, add_special_tokens=True).input_ids
        if len(encoded) == target_tokens:
            return prompt
    raise RuntimeError(
        "cannot construct an exact {}-token prompt with tokenizer; provide a runner adaptation rather than silently changing the case".format(target_tokens)
    )


def build_command(args, case, output_path, prompt):
    command = [
        str(args.python), str(ROOT / "tools" / "run_llama31.py"),
        "--checkpoint", str(args.checkpoint), "--prompt", prompt,
        "--max-new-tokens", str(case["decode_tokens"]),
        "--device", str(args.device), "--weight-store", args.weight_store,
        "--granularity", args.granularity, "--slots", str(args.slots),
        "--prefetch-depth", str(args.prefetch_depth),
        "--embedding-placement", args.embedding_placement,
        "--lm-head-placement", args.lm_head_placement,
        "--kv-page-size", str(args.kv_page_size),
        "--kv-attention-backend", args.kv_attention_backend,
        "--kv-prefill-backend", args.kv_prefill_backend,
        "--cuda-safety-margin-mib", str(args.cuda_safety_margin_mib),
        "--gpu-resident-weight-budget", str(args.gpu_resident_weight_budget),
        "--output", str(output_path),
    ]
    if args.allow_kv_reference:
        command.append("--allow-kv-reference")
    if args.ignore_memlock_limit:
        command.append("--ignore-memlock-limit")
    return command


def display_command(command, prompt_tokens):
    """Keep reproducible policy without embedding local paths or huge prompts."""

    result = list(command)
    try:
        result[result.index("--checkpoint") + 1] = "$CASCADE_70B_CHECKPOINT"
    except (ValueError, IndexError):
        pass
    try:
        result[result.index("--prompt") + 1] = "<generated_exact_{}_token_prompt>".format(
            prompt_tokens
        )
    except (ValueError, IndexError):
        pass
    return result


def _blocked_case(case, status, reason, evidence_class):
    return dict(case, status=status, evidence_class=evidence_class, reason=reason, command=None, metrics=None)


def compact_admission(environment):
    return {
        "captured_at": environment.get("captured_at"),
        "selected_physical_gpu": environment.get("selected_physical_gpu"),
        "selected_gpu_external_compute_processes": environment.get(
            "selected_gpu_external_compute_processes"
        ),
        "exclusive_snapshot": environment.get("exclusive_snapshot"),
        "nvidia_smi_errors": environment.get("nvidia_smi_errors", []),
    }


def _redact_checkpoint(value, checkpoint):
    return str(value).replace(str(checkpoint), "$CASCADE_70B_CHECKPOINT")


def execute_case(args, case, evidence_class, admission_environment=None):
    started = time.perf_counter()
    raw_dir = args.raw_run_dir / case["case_id"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path = raw_dir / "run_report.json"
    stdout_path = raw_dir / "stdout.log"
    stderr_path = raw_dir / "stderr.log"
    process = None
    monitor = None
    monitor_result = None
    try:
        prompt = _build_prompt(args.checkpoint, case["prompt_tokens"])
        command = build_command(args, case, report_path, prompt)
        selected = (admission_environment or {}).get("selected_physical_gpu")
        with stdout_path.open("w", encoding="utf-8") as stdout_stream, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_stream:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
            )
            if selected is not None:
                monitor = GPUActivityMonitor(selected["uuid"])
                monitor.allow_pid(process.pid)
                monitor.start()
            try:
                returncode = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise TimeoutError(
                    "run_llama31 exceeded {} seconds".format(
                        args.timeout_seconds
                    )
                ) from error
            finally:
                if monitor is not None:
                    monitor_result = monitor.stop()
        if returncode != 0:
            raise RuntimeError(
                "run_llama31 exited {} (see {})".format(
                    returncode, stderr_path
                )
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        actual_prompt = _nested(report, "throughput", "prompt_tokens")
        actual_decode = _nested(report, "throughput", "generated_tokens")
        if actual_prompt != case["prompt_tokens"] or actual_decode != case["decode_tokens"]:
            raise RuntimeError(
                "case shape mismatch: expected {}/{}, got {}/{}".format(
                    case["prompt_tokens"], case["decode_tokens"], actual_prompt, actual_decode
                )
            )
        return dict(
            case, status=PASS, evidence_class=evidence_class, reason=None,
            duration_seconds=time.perf_counter() - started,
            command=display_command(command, case["prompt_tokens"]),
            raw_report=str(report_path), metrics=normalize_run_report(report),
            gpu_activity_monitor=monitor_result,
        )
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if monitor is not None and monitor_result is None:
            monitor_result = monitor.stop()
        return dict(
            case, status=FAIL, evidence_class=evidence_class,
            reason=_redact_checkpoint(
                "{}: {}".format(type(error).__name__, error), args.checkpoint
            ),
            duration_seconds=time.perf_counter() - started,
            traceback=_redact_checkpoint(traceback.format_exc(), args.checkpoint),
            raw_report=str(report_path) if report_path.exists() else None,
            gpu_activity_monitor=monitor_result,
        )


def render_report(summary, cases):
    lines = [
        "# Llama 70B Dense KV Single-Request Qualification",
        "",
        "Status: `{}`  ".format(summary["status"]),
        "Evidence class: `{}`  ".format(summary["evidence_class"]),
        "Mode: `{}`".format(summary["mode"]),
        "",
        summary["qualification_statement"],
        "",
        "| Case | Prompt | Decode | Result | Evidence |",
        "|---|---:|---:|---|---|",
    ]
    for case in cases:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                case["case_id"], case["prompt_tokens"], case["decode_tokens"],
                case["status"], case["evidence_class"],
            )
        )
    lines.extend(
        [
            "",
            "Qualification requires scheduler-backed GPU allocation evidence plus per-case start/end process snapshots; smoke remains non-qualifying.",
            "Unavailable child-process resources retain an explicit source marker and are never inferred.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    cases = planned_cases(args.cases, args.repeat)
    environment = capture_cuda_environment(ROOT, args.device)
    environment["schema_version"] = SCHEMA_VERSION
    environment["checkpoint"] = (
        None
        if args.checkpoint is None
        else {"name": args.checkpoint.name, "local_path_redacted": True}
    )
    environment["runner"] = str(Path(__file__).relative_to(ROOT))
    generated_at = utc_now()
    run_id = generated_at.replace(":", "").replace("+", "_")
    args.raw_run_dir = args.raw_dir / run_id
    if args.mode == "plan":
        evidence_class = PLAN_ONLY
        results = [
            _blocked_case(case, SKIPPED, "PLAN_ONLY: checkpoint and CUDA execution intentionally skipped", evidence_class)
            for case in cases
        ]
        status = PLAN_ONLY
        statement = "This bundle is an execution plan, not model or CUDA qualification evidence."
    else:
        admitted, admission_status, reason = qualification_admission(
            environment,
            args.allow_shared_smoke and not args.require_uncontended_benchmark,
            require_reservation=(
                args.mode == "qualification" and not args.allow_shared_smoke
            ),
        )
        external = int(environment.get("selected_gpu_external_compute_processes", 0))
        shared = args.mode == "smoke" or args.allow_shared_smoke or external > 0
        evidence_class = SMOKE_ONLY if shared else "QUALIFICATION"
        if not admitted:
            status = (
                BLOCKED_NOT_EXCLUSIVE
                if reason and BLOCKED_NOT_EXCLUSIVE in reason
                else admission_status
            )
            results = [_blocked_case(case, status, reason, evidence_class) for case in cases]
            statement = "No model case executed because CUDA admission failed."
        else:
            results = []
            for case in cases:
                fresh = capture_cuda_environment(ROOT, args.device)
                fresh_admitted, fresh_status, fresh_reason = qualification_admission(
                    fresh,
                    args.allow_shared_smoke
                    and not args.require_uncontended_benchmark,
                    require_reservation=(
                        args.mode == "qualification"
                        and not args.allow_shared_smoke
                    ),
                )
                if not fresh_admitted:
                    case_status = (
                        BLOCKED_NOT_EXCLUSIVE
                        if fresh_reason and BLOCKED_NOT_EXCLUSIVE in fresh_reason
                        else fresh_status
                    )
                    results.append(_blocked_case(case, case_status, fresh_reason, evidence_class))
                    continue
                result = execute_case(
                    args, case, evidence_class, admission_environment=fresh
                )
                after = capture_cuda_environment(ROOT, args.device)
                result["gpu_admission"] = {
                    "before": compact_admission(fresh),
                    "after": compact_admission(after),
                    "note": "scheduler reservation plus point-in-time process snapshots are required for qualification",
                }
                if (
                    evidence_class == "QUALIFICATION"
                    and int(after.get("selected_gpu_external_compute_processes", 0)) > 0
                ):
                    result["status"] = FAIL
                    result["reason"] = (
                        "exclusive GPU condition was lost before the post-case snapshot"
                    )
                monitor_external = bool(
                    (result.get("gpu_activity_monitor") or {}).get(
                        "external_activity_observed"
                    )
                )
                if args.require_uncontended_benchmark and (
                    int(after.get("selected_gpu_external_compute_processes", 0))
                    or monitor_external
                ):
                    result["status"] = FAIL
                    result["reason"] = "INVALIDATED_EXTERNAL_GPU_ACTIVITY"
                results.append(result)
            any_fail = any(case["status"] == FAIL for case in results)
            any_block = any(case["status"] in {"BLOCKED", BLOCKED_NOT_EXCLUSIVE} for case in results)
            if any_fail:
                status = FAIL
            elif any_block:
                status = BLOCKED_NOT_EXCLUSIVE
            elif evidence_class == SMOKE_ONLY:
                status = SMOKE_ONLY
            else:
                status = PASS
            statement = (
                "Shared/smoke execution is diagnostic only and cannot qualify the 70B path."
                if evidence_class == SMOKE_ONLY
                else "PASS is qualification evidence only when all cases ran inside the admitted scheduler GPU allocation and retained clean process snapshots."
            )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "mode": args.mode,
        "status": status,
        "evidence_class": evidence_class,
        "qualification_statement": statement,
        "case_count": len(results),
        "pass_count": sum(case["status"] == PASS for case in results),
        "fail_count": sum(case["status"] == FAIL for case in results),
        "not_run_count": sum(case["status"] != PASS and case["status"] != FAIL for case in results),
        "matrix": [[case["prompt_tokens"], case["decode_tokens"]] for case in cases],
        "matrix_coverage": {
            "required_prompt_lengths": list(REQUIRED_PROMPT_LENGTHS),
            "required_decode_lengths": list(REQUIRED_DECODE_LENGTHS),
            "covered_prompt_lengths": sorted({case["prompt_tokens"] for case in cases}),
            "covered_decode_lengths": sorted({case["decode_tokens"] for case in cases}),
        },
        "runner_config": {
            name: getattr(args, name)
            for name in (
                "device", "weight_store", "granularity", "slots", "prefetch_depth",
                "embedding_placement", "lm_head_placement", "kv_page_size",
                "kv_attention_backend", "kv_prefill_backend", "allow_shared_smoke",
                "require_uncontended_benchmark",
            )
        } | {
            "gpu_resident_weight_budget": args.gpu_resident_weight_budget,
            "raw_run_dir": str(args.raw_run_dir),
        },
    }
    markdown = render_report(summary, results)
    paths = write_report_bundle(args.output_dir, environment, results, summary, markdown)
    return summary, results, paths


def main(argv=None):
    args = parse_args(argv)
    summary, _, paths = run(args)
    print(json.dumps({"summary": summary, "reports": paths}, indent=2, sort_keys=True))
    return 0 if summary["status"] in {PASS, SMOKE_ONLY, PLAN_ONLY} else 1


if __name__ == "__main__":
    raise SystemExit(main())
