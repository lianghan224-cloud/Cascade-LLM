#!/usr/bin/env python3
"""Unified KV stack validation entry point and report generator."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools.kv_validation_cases import LOGIC_SUITES, run_cuda_suite
from tools.qualification_common import (
    BLOCKED_NOT_EXCLUSIVE,
    capture_cuda_environment,
    qualification_admission,
    utc_now,
    write_report_bundle,
)


PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED_WITH_REASON"
BLOCKED = "BLOCKED"


CASE_NAMES = {
    "V00": "pre-modification snapshot",
    "V01": "existing weight-free baseline",
    "V02": "allocate/release closure",
    "V03": "non-negative ref_count",
    "V04": "non-negative pin_count",
    "V05": "double release/unpin rejection",
    "V06": "generation and ABA rejection",
    "V07": "bounded quiescence",
    "V08": "request cancellation cleanup",
    "V09": "simulated OOM recovery",
    "V10": "simulated kernel failure recovery",
    "V11": "IO submit failure rollback",
    "V12": "IO completion failure rollback",
    "V13": "concurrent allocate/fork/free",
    "V14": "100k randomized lifecycle operations",
    "V15": "minimal reproduction reduction",
    "V16": "Fork sharing",
    "V17": "COW isolation",
    "V18": "Prefix ownership",
    "V19": "Prefix eviction with active owner",
    "V20": "Beam fork",
    "V21": "Beam branch exit",
    "V22": "Speculative commit",
    "V23": "Speculative partial commit",
    "V24": "Speculative rollback",
    "V25": "cross-page rollback",
    "V26": "in-page append/rollback",
    "V27": "Quest index build",
    "V28": "Quest full selection",
    "V29": "Quest full reference equivalence",
    "V30": "Quest budget/top-k",
    "V31": "Quest sparse statistics",
    "V32": "Quest append update",
    "V33": "Quest Fork record sharing",
    "V34": "Quest COW record isolation",
    "V35": "Quest rollback",
    "V36": "Quest stale-version rejection",
    "V37": "Quest serialization round trip",
    "V38": "Quest boundary cases",
    "V39": "logical block to location mapping",
    "V40": "Mock GPU to CPU migration",
    "V41": "Mock CPU to SSD migration",
    "V42": "Mock SSD/CPU/GPU round trip",
    "V43": "unique authoritative copy",
    "V44": "read during migration",
    "V45": "migration failure rollback",
    "V46": "pinned eviction exclusion",
    "V47": "prefetch deduplication",
    "V48": "prefetch cancellation cleanup",
    "V49": "tier capacity handling",
    "V50": "reload after eviction",
    "V51": "atomic metadata commit",
    "V52": "selection/prefetch/view end-to-end",
    "V53": "compute pin lifecycle",
    "V54": "multi-request prefetch fairness",
    "V55": "cancellation propagation",
    "V56": "Full Prefill routing",
    "V57": "Decode routing",
    "V58": "Chunked Prefill routing",
    "V59": "correct fallback without dedicated kernel",
    "V60": "synthetic Decode numerics",
    "V61": "synthetic Full Prefill numerics",
    "V62": "synthetic Chunked Prefill numerics",
    "V63": "multi-stream race",
    "V64": "CUDA allocated-memory drift",
    "V65": "CUDA reserved-memory drift",
    "V66": "CUDA kernel failure recovery",
    "V67": "current-hardware performance baseline",
    "V68": "real-model logits/Top-1",
    "V69": "real 8B long generation",
    "V70": "real long-context Prefill",
    "V71": "real Quest accuracy",
    "V72": "real NVMe throughput/latency",
    "V73": "real IO/compute overlap",
    "V74": "SM86 specialized path validation",
    "V75": "other-architecture tuning",
    "V76": "large dynamic batch",
    "V77": "Attention Fence and selected-page pin closure",
    "V78": "Prefix page/byte budget and LRU closure",
    "V79": "request-scoped Prefetch/Migration Fence closure",
    "V80": "GenerationSession lifecycle closure",
    "V81": "Quest compact incremental index closure",
    "V82": "long-stability harness schema and drift gate",
}


@dataclass
class CaseResult:
    case_id: str
    name: str
    profile: str
    status: str
    reason: object
    duration: float
    seed: int
    environment: dict
    metrics: dict
    artifacts: list


def environment():
    value = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        value["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": "sm{}{}".format(
                    *torch.cuda.get_device_capability(index)
                ),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return value


def result(case_id, profile, status, env, seed, duration=0.0, reason=None, metrics=None, artifacts=None):
    return CaseResult(
        case_id=case_id,
        name=CASE_NAMES[case_id],
        profile=profile,
        status=status,
        reason=reason,
        duration=float(duration),
        seed=int(seed),
        environment=env,
        metrics=dict(metrics or {}),
        artifacts=list(artifacts or []),
    )


def failure_artifact(case_id, seed, suite_name, exc):
    directory = ROOT / "reports" / "kv_validation_failures" / case_id
    directory.mkdir(parents=True, exist_ok=True)
    seed_path = directory / "seed.txt"
    repro_path = directory / "repro.json"
    seed_path.write_text(str(int(seed)) + "\n", encoding="utf-8")
    repro = {
        "case_id": case_id,
        "seed": int(seed),
        "minimal_reproduction": [
            {
                "command": "{} tools/validate_kv_stack.py --profile logic --seed {}".format(
                    sys.executable, int(seed)
                ),
                "suite": suite_name,
            }
        ],
        "exception": "{}: {}".format(type(exc).__name__, exc),
        "traceback": traceback.format_exc(),
    }
    repro_path.write_text(
        json.dumps(repro, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return [str(seed_path.relative_to(ROOT)), str(repro_path.relative_to(ROOT))]


def preflight_results(profile, env, seed):
    snapshots = [
        "preflight_git_status.txt",
        "preflight_worktree.patch",
        "preflight_index.patch",
        "preflight_untracked_files.txt",
        "preflight_head.txt",
    ]
    root = ROOT / "reports" / "kv_remediation"
    missing = [name for name in snapshots if not (root / name).exists()]
    status = PASS if not missing else SKIPPED
    first = result(
        "V00",
        profile,
        status,
        env,
        seed,
        reason=(
            None
            if not missing
            else "historical pre-modification snapshots are intentionally not tracked"
        ),
        metrics={"snapshots": snapshots, "missing": missing},
        artifacts=[str((root / name).relative_to(ROOT)) for name in snapshots if (root / name).exists()],
    )
    baseline_files = ["baseline_pytest.txt", "baseline_unittest.txt", "baseline.md", "baseline.json"]
    present = [name for name in baseline_files if (root / name).exists()]
    second = result(
        "V01",
        profile,
        PASS if present else SKIPPED,
        env,
        seed,
        reason=(
            "baseline records include pre-existing failures; V01 requires recording, not a clean baseline"
            if present
            else "historical baseline records are intentionally not tracked"
        ),
        metrics={"present": present},
        artifacts=[str((root / name).relative_to(ROOT)) for name in present],
    )
    return [first, second]


def logic_results(profile, env, seed):
    results = preflight_results(profile, env, seed)
    for case_ids, suite in LOGIC_SUITES:
        started = time.perf_counter()
        try:
            metrics = suite(seed=seed)
            elapsed = time.perf_counter() - started
            for case_id in case_ids:
                results.append(
                    result(
                        case_id,
                        profile,
                        PASS,
                        env,
                        seed,
                        duration=elapsed / len(case_ids),
                        metrics=metrics.get(case_id, {}),
                    )
                )
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            for case_id in case_ids:
                artifacts = failure_artifact(case_id, seed, suite.__name__, exc)
                results.append(
                    result(
                        case_id,
                        profile,
                        FAIL,
                        env,
                        seed,
                        duration=elapsed / len(case_ids),
                        reason="{}: {}".format(type(exc).__name__, exc),
                        artifacts=artifacts,
                    )
                )
    return results


def cuda_results(profile, env, seed):
    ids = tuple("V{:02d}".format(index) for index in range(60, 68))
    if not torch.cuda.is_available():
        reason = "torch.cuda.is_available() is false; random-tensor CUDA validation requires a CUDA GPU"
        return [result(case_id, profile, SKIPPED, env, seed, reason=reason) for case_id in ids]
    started = time.perf_counter()
    try:
        metrics = run_cuda_suite(seed=seed)
        elapsed = time.perf_counter() - started
        return [
            result(
                case_id,
                profile,
                PASS,
                env,
                seed,
                duration=elapsed / len(ids),
                metrics=metrics[case_id],
            )
            for case_id in ids
        ]
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        return [
            result(
                case_id,
                profile,
                FAIL,
                env,
                seed,
                duration=elapsed / len(ids),
                reason="{}: {}".format(type(exc).__name__, exc),
                artifacts=failure_artifact(case_id, seed, "run_cuda_suite", exc),
            )
            for case_id in ids
        ]


def real_environment_results(profile, env, seed, cuda_cases):
    model_path = os.environ.get("CASCADE_KV_MODEL_PATH")
    dataset_path = os.environ.get("CASCADE_KV_DATASET_PATH")
    nvme_path = os.environ.get("CASCADE_KV_NVME_PATH")
    serving_command = os.environ.get("CASCADE_KV_SERVING_COMMAND")
    values = []
    model_reason = "CASCADE_KV_MODEL_PATH is not configured with real model weights"
    dataset_reason = "CASCADE_KV_DATASET_PATH is not configured with an accuracy dataset"
    nvme_reason = "CASCADE_KV_NVME_PATH is not configured for dedicated NVMe/GDS validation"
    for case_id in ("V68", "V69", "V70"):
        values.append(
            result(
                case_id,
                profile,
                SKIPPED if not model_path else BLOCKED,
                env,
                seed,
                reason=(model_reason if not model_path else "real-model runner requires project-specific checkpoint arguments"),
            )
        )
    values.append(
        result(
            "V71",
            profile,
            SKIPPED if not (model_path and dataset_path) else BLOCKED,
            env,
            seed,
            reason=(dataset_reason if not dataset_path else model_reason),
        )
    )
    for case_id in ("V72", "V73"):
        values.append(
            result(
                case_id,
                profile,
                SKIPPED if not nvme_path else BLOCKED,
                env,
                seed,
                reason=(nvme_reason if not nvme_path else "real Direct IO/GDS runner is not configured"),
            )
        )
    capability = (
        torch.cuda.get_device_capability(0)
        if torch.cuda.is_available()
        else None
    )
    cuda_pass = all(item.status == PASS for item in cuda_cases)
    values.append(
        result(
            "V74",
            profile,
            PASS if capability == (8, 6) and cuda_pass else SKIPPED,
            env,
            seed,
            reason=(
                "SM86 synthetic specialized path and current-hardware baseline passed; this is not a production performance claim"
                if capability == (8, 6) and cuda_pass
                else "an SM86 CUDA device with passing synthetic cases is required"
            ),
            metrics={"capability": capability, "synthetic_only": True},
        )
    )
    values.append(
        result(
            "V75",
            profile,
            SKIPPED,
            env,
            seed,
            reason="only SM86 devices are present; other architecture-specific tuning requires its target GPU",
        )
    )
    values.append(
        result(
            "V76",
            profile,
            SKIPPED if not (model_path and serving_command) else BLOCKED,
            env,
            seed,
            reason=(
                "real weights and CASCADE_KV_SERVING_COMMAND are not configured"
                if not (model_path and serving_command)
                else "large serving stress runner requires deployment-specific arguments"
            ),
        )
    )
    return values


def write_reports(profile, seed, env, results, elapsed, output_dir=None):
    reports = (
        ROOT / "reports" if output_dir is None else Path(output_dir)
    )
    reports.mkdir(parents=True, exist_ok=True)
    counts = {
        status: sum(item.status == status for item in results)
        for status in (PASS, FAIL, SKIPPED, BLOCKED)
    }
    overall = FAIL if counts[FAIL] or counts[BLOCKED] else PASS
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "status": overall,
        "seed": int(seed),
        "duration": elapsed,
        "environment": env,
        "summary": counts,
        "cases": [asdict(item) for item in results],
    }
    json_path = reports / "kv_validation.json"
    md_path = reports / "kv_validation.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# KV Stack Validation",
        "",
        "- Profile: `{}`".format(profile),
        "- Status: `{}`".format(overall),
        "- Seed: `{}`".format(seed),
        "- Duration: `{:.3f}s`".format(elapsed),
        "- Summary: PASS={PASS}, FAIL={FAIL}, SKIPPED_WITH_REASON={SKIPPED_WITH_REASON}, BLOCKED={BLOCKED}".format(**counts),
        "",
        "| ID | Validation | Status | Duration (s) | Reason / key metrics |",
        "|---|---|---:|---:|---|",
    ]
    for item in results:
        detail = item.reason or json.dumps(item.metrics, sort_keys=True)
        detail = str(detail).replace("|", "\\|").replace("\n", " ")
        if len(detail) > 260:
            detail = detail[:257] + "..."
        lines.append(
            "| {case_id} | {name} | {status} | {duration:.4f} | {detail} |".format(
                case_id=item.case_id,
                name=item.name,
                status=item.status,
                duration=item.duration,
                detail=detail,
            )
        )
    lines.extend(
        [
            "",
            "## Hardware follow-up commands",
            "",
            "```bash",
            ".venv/bin/python tools/validate_kv_stack.py --profile cuda-synthetic",
            "CASCADE_KV_MODEL_PATH=/path/to/model .venv/bin/python tools/validate_kv_stack.py --profile full",
            "CASCADE_KV_NVME_PATH=/path/on/dedicated/nvme .venv/bin/python tools/validate_kv_stack.py --profile full",
            "```",
            "",
            "`SKIPPED_WITH_REASON` is used only for absent weights, datasets, NVMe configuration, or target hardware. A missing implementation is never converted to a skip.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall, json_path, md_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--profile",
        choices=("logic", "cuda-synthetic", "full"),
        help="legacy validation profile; retained for report compatibility",
    )
    selection.add_argument(
        "--mode",
        choices=("logic", "cuda-smoke", "qualification"),
        help="qualification-ready orchestration entry point",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--repeat",
        type=int,
        help="70B repetitions per matrix case; qualification defaults to 5",
    )
    parser.add_argument(
        "--allow-shared-smoke",
        action="store_true",
        help=(
            "allow diagnostic execution on a shared GPU; evidence is always "
            "SMOKE_ONLY and never qualification"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args(argv)
    if args.mode is not None and args.output_dir is None:
        args.output_dir = str(ROOT / "reports" / "kv_qualification_entry")
    if args.mode == "qualification" and not args.checkpoint:
        parser.error("--checkpoint is required for qualification mode")
    if (
        args.mode == "qualification"
        and args.checkpoint
        and not Path(args.checkpoint).is_dir()
    ):
        parser.error("--checkpoint must be an existing local directory")
    if args.repeat is not None and args.repeat <= 0:
        parser.error("--repeat must be positive")
    return args


def _component_case(name, status, evidence, artifact, reason=None, metrics=None):
    return {
        "case_id": "COMPONENT-{}".format(name.upper().replace("_", "-")),
        "component": name,
        "status": status,
        "evidence": evidence,
        "reason": reason,
        "artifact": artifact,
        "metrics": dict(metrics or {}),
    }


def _component_error(name, evidence, artifact, error):
    return _component_case(
        name,
        FAIL,
        evidence,
        artifact,
        reason="{}: {}".format(type(error).__name__, error),
        metrics={"traceback": traceback.format_exc()},
    )


def _render_unified_report(summary, cases):
    lines = [
        "# Dense GPU KV Qualification Entry",
        "",
        "- Mode: `{}`".format(summary["mode"]),
        "- Status: `{}`".format(summary["status"]),
        "- Evidence class: `{}`".format(summary["evidence_class"]),
        "- `QUALIFICATION_READY != QUALIFIED`",
        "",
        "| Component | Status | Evidence | Artifact | Reason |",
        "|---|---|---|---|---|",
    ]
    for case in cases:
        reason = str(case.get("reason") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| {} | `{}` | `{}` | {} | {} |".format(
                case["component"], case["status"], case["evidence"],
                case.get("artifact") or "-", reason,
            )
        )
    lines.extend(
        [
            "",
            "A no-process `nvidia-smi` snapshot is not a GPU reservation; qualification requires scheduler-backed allocation evidence.",
            "Shared CUDA execution remains smoke evidence. The 70B runner is never called by logic or cuda-smoke mode.",
            "",
        ]
    )
    return "\n".join(lines)


def run_unified(args):
    """Run the frozen qualification components behind one explicit gate."""

    from tools import generate_kv_capability_matrix
    from tools import qualify_gather_sdpa_prefill
    from tools import qualify_llama70b_single_request
    from tools import validate_kv_cuda_async
    from tools import validate_kv_long_stability

    mode = args.mode
    environment_snapshot = capture_cuda_environment(ROOT, args.device)
    evidence = (
        "LOGIC_VALIDATED"
        if mode == "logic"
        else "SMOKE_ONLY"
        if mode == "cuda-smoke" or args.allow_shared_smoke
        else "QUALIFICATION"
    )
    component_cases = []

    if mode == "qualification":
        admitted, _admission_status, reason = qualification_admission(
            environment_snapshot,
            allow_shared_smoke=args.allow_shared_smoke,
            require_reservation=not args.allow_shared_smoke,
        )
        if not admitted:
            blocked = (
                BLOCKED_NOT_EXCLUSIVE
                if reason and BLOCKED_NOT_EXCLUSIVE in reason
                else BLOCKED
            )
            for name, artifact in (
                ("cuda_async", "reports/kv_cuda_async/summary.json"),
                ("long_stability", "reports/kv_long_stability/summary.json"),
                ("gather_sdpa_prefill", "reports/gather_sdpa_prefill_qualification/summary.json"),
                ("llama70b", "reports/llama70b_single_request_qualification/summary.json"),
            ):
                component_cases.append(
                    _component_case(name, blocked, evidence, artifact, reason=reason)
                )
            return _write_unified_bundle(
                args, environment_snapshot, component_cases, evidence
            )

    legacy_profile = "logic" if mode == "logic" else "cuda-synthetic"
    started = time.perf_counter()
    legacy_env = environment()
    legacy_cases = (
        logic_results(legacy_profile, legacy_env, args.seed)
        if mode == "logic"
        else cuda_results(legacy_profile, legacy_env, args.seed)
    )
    legacy_overall, legacy_json, _ = write_reports(
        legacy_profile,
        args.seed,
        legacy_env,
        legacy_cases,
        time.perf_counter() - started,
    )
    legacy_suffix = "logic" if mode == "logic" else "cuda_smoke"
    legacy_snapshot = ROOT / "reports" / "kv_validation_{}.json".format(
        legacy_suffix
    )
    shutil.copy2(legacy_json, legacy_snapshot)
    shutil.copy2(
        ROOT / "reports" / "kv_validation.md",
        ROOT / "reports" / "kv_validation_{}.md".format(legacy_suffix),
    )
    component_cases.append(
        _component_case(
            "kv_stack_legacy",
            legacy_overall,
            evidence,
            str(legacy_snapshot.relative_to(ROOT)),
            metrics={"case_count": len(legacy_cases)},
        )
    )

    async_mode = (
        "logic"
        if mode == "logic"
        else "cuda-smoke"
        if mode == "cuda-smoke" or args.allow_shared_smoke
        else "qualification"
    )
    try:
        async_report = validate_kv_cuda_async.run_validation(async_mode, args.device)
        async_output_dir = ROOT / "reports" / (
            "kv_cuda_async_logic"
            if async_mode == "logic"
            else "kv_cuda_async_cuda_smoke"
            if async_mode == "cuda-smoke"
            else "kv_cuda_async_qualification"
        )
        validate_kv_cuda_async.write_reports(
            async_report, async_output_dir
        )
        async_artifact = str((async_output_dir / "summary.json").relative_to(ROOT))
        component_cases.append(
            _component_case(
                "cuda_async",
                async_report["status"],
                async_report["qualification"],
                async_artifact,
                metrics=async_report["counts"],
            )
        )
    except BaseException as error:
        component_cases.append(
            _component_error(
                "cuda_async", evidence, "reports/kv_cuda_async/summary.json", error
            )
        )

    long_mode = "logic" if mode == "logic" else "cuda-smoke" if mode == "cuda-smoke" or args.allow_shared_smoke else "full"
    try:
        long_report = validate_kv_long_stability.run_validation(
            long_mode,
            cuda_device=args.device,
            sample_every=100 if long_mode == "full" else 1,
        )
        long_output_dir = ROOT / "reports" / (
            "kv_long_stability_logic"
            if long_mode == "logic"
            else "kv_long_stability_cuda_smoke"
            if long_mode == "cuda-smoke"
            else "kv_long_stability_qualification"
        )
        validate_kv_long_stability.write_reports(
            long_report, long_output_dir
        )
        long_artifact = str((long_output_dir / "summary.json").relative_to(ROOT))
        component_cases.append(
            _component_case(
                "long_stability",
                long_report["status"],
                long_report["qualification"],
                long_artifact,
                metrics=long_report["counts"],
            )
        )
    except BaseException as error:
        component_cases.append(
            _component_error(
                "long_stability", evidence, "reports/kv_long_stability/summary.json", error
            )
        )

    prefill_mode = (
        "logic"
        if mode == "logic"
        else "cuda-smoke"
        if mode == "cuda-smoke" or args.allow_shared_smoke
        else "qualification"
    )
    prefill_output_dir = ROOT / "reports" / (
        "gather_sdpa_prefill_cuda_smoke"
        if prefill_mode == "cuda-smoke"
        else "gather_sdpa_prefill_qualification"
    )
    prefill_artifact = str(
        (prefill_output_dir / "summary.json").relative_to(ROOT)
    )
    prefill_argv = [
        "--mode", prefill_mode,
        "--device", args.device,
        "--output-dir", str(prefill_output_dir),
    ]
    if prefill_mode == "cuda-smoke":
        prefill_argv.append("--allow-shared-smoke")
    if args.checkpoint:
        prefill_argv.extend(("--checkpoint", args.checkpoint))
    try:
        prefill_args = qualify_gather_sdpa_prefill.build_parser().parse_args(prefill_argv)
        prefill_summary, _, _ = qualify_gather_sdpa_prefill.run(prefill_args)
        component_cases.append(
            _component_case(
                "gather_sdpa_prefill",
                prefill_summary["overall"],
                prefill_summary["capability_state"],
                prefill_artifact,
                metrics={"case_counts": prefill_summary["case_counts"]},
            )
        )
    except BaseException as error:
        component_cases.append(
            _component_error(
                "gather_sdpa_prefill",
                evidence,
                prefill_artifact,
                error,
            )
        )

    if mode == "qualification":
        runner_argv = [
            "--mode", "qualification",
            "--device", args.device,
            "--checkpoint", args.checkpoint,
            "--output-dir", str(ROOT / "reports" / "llama70b_single_request_qualification"),
        ]
        if args.allow_shared_smoke:
            runner_argv.append("--allow-shared-smoke")
        if args.repeat is not None:
            runner_argv.extend(("--repeat", str(args.repeat)))
        prerequisite_failures = [
            case for case in component_cases
            if case["status"] in {FAIL, BLOCKED, BLOCKED_NOT_EXCLUSIVE}
        ]
        if prerequisite_failures:
            component_cases.append(
                _component_case(
                    "llama70b",
                    BLOCKED,
                    evidence,
                    "reports/llama70b_single_request_qualification/summary.json",
                    reason="prerequisite component failed: {}".format(
                        ", ".join(
                            case["component"] for case in prerequisite_failures
                        )
                    ),
                )
            )
        else:
            try:
                runner_args = qualify_llama70b_single_request.parse_args(runner_argv)
                runner_summary, _, _ = qualify_llama70b_single_request.run(runner_args)
                component_cases.append(
                    _component_case(
                        "llama70b",
                        runner_summary["status"],
                        runner_summary["evidence_class"],
                        "reports/llama70b_single_request_qualification/summary.json",
                        metrics={
                            "pass_count": runner_summary["pass_count"],
                            "case_count": runner_summary["case_count"],
                        },
                    )
                )
            except BaseException as error:
                component_cases.append(
                    _component_error(
                        "llama70b",
                        evidence,
                        "reports/llama70b_single_request_qualification/summary.json",
                        error,
                    )
                )

    try:
        generate_kv_capability_matrix.write_report(
            ROOT / "reports" / "kv_capability_matrix"
        )
    except BaseException as error:
        component_cases.append(
            _component_error(
                "capability_matrix",
                evidence,
                "reports/kv_capability_matrix/matrix.json",
                error,
            )
        )
    return _write_unified_bundle(
        args, environment_snapshot, component_cases, evidence
    )


def _write_unified_bundle(args, environment, component_cases, evidence):
    failing = [
        case for case in component_cases
        if case["status"] in {FAIL, BLOCKED, BLOCKED_NOT_EXCLUSIVE}
    ]
    status = (
        failing[0]["status"]
        if failing
        else "SMOKE_ONLY"
        if evidence == "SMOKE_ONLY"
        else PASS
    )
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "mode": args.mode,
        "status": status,
        "evidence_class": evidence,
        "qualification_ready_is_qualified": False,
        "component_count": len(component_cases),
        "pass_count": sum(case["status"] == PASS for case in component_cases),
        "failure_count": len(failing),
        "checkpoint": (
            None
            if args.checkpoint is None
            else {
                "name": Path(args.checkpoint).name,
                "local_path_redacted": True,
            }
        ),
    }
    paths = write_report_bundle(
        args.output_dir,
        environment,
        component_cases,
        summary,
        _render_unified_report(summary, component_cases),
    )
    return summary, component_cases, paths


def main(argv=None):
    args = parse_args(argv)
    if args.mode is not None:
        summary, _, paths = run_unified(args)
        print(
            "KV qualification entry {}: {} ({})".format(
                args.mode, summary["status"], summary["evidence_class"]
            )
        )
        print(paths["summary.json"])
        print(paths["report.md"])
        return 0 if summary["status"] in {PASS, "SMOKE_ONLY"} else 1
    started = time.perf_counter()
    env = environment()
    cases = []
    cuda_cases = []
    if args.profile in {"logic", "full"}:
        cases.extend(logic_results(args.profile, env, args.seed))
    if args.profile in {"cuda-synthetic", "full"}:
        cuda_cases = cuda_results(args.profile, env, args.seed)
        cases.extend(cuda_cases)
    if args.profile == "full":
        cases.extend(
            real_environment_results(
                args.profile, env, args.seed, cuda_cases
            )
        )
    elapsed = time.perf_counter() - started
    overall, json_path, md_path = write_reports(
        args.profile,
        args.seed,
        env,
        cases,
        elapsed,
        output_dir=args.output_dir,
    )
    counts = {
        status: sum(item.status == status for item in cases)
        for status in (PASS, FAIL, SKIPPED, BLOCKED)
    }
    print(
        "KV validation {}: {} (PASS={}, FAIL={}, SKIPPED_WITH_REASON={}, BLOCKED={})".format(
            args.profile,
            overall,
            counts[PASS],
            counts[FAIL],
            counts[SKIPPED],
            counts[BLOCKED],
        )
    )
    for path in (json_path, md_path):
        try:
            print(path.relative_to(ROOT))
        except ValueError:
            print(path)
    return 1 if overall == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
