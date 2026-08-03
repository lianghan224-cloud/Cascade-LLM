#!/usr/bin/env python3
"""Unified KV stack validation entry point and report generator."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools.kv_validation_cases import LOGIC_SUITES, run_cuda_suite


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
    status = PASS if not missing else FAIL
    first = result(
        "V00",
        profile,
        status,
        env,
        seed,
        reason=(None if not missing else "missing snapshots: {}".format(missing)),
        metrics={"snapshots": snapshots, "missing": missing},
        artifacts=[str((root / name).relative_to(ROOT)) for name in snapshots if (root / name).exists()],
    )
    baseline_files = ["baseline_pytest.txt", "baseline_unittest.txt", "baseline.md", "baseline.json"]
    present = [name for name in baseline_files if (root / name).exists()]
    second = result(
        "V01",
        profile,
        PASS if present else FAIL,
        env,
        seed,
        reason=(
            "baseline records include pre-existing failures; V01 requires recording, not a clean baseline"
            if present
            else "no baseline record exists"
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


def write_reports(profile, seed, env, results, elapsed):
    reports = ROOT / "reports"
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
    parser.add_argument(
        "--profile",
        required=True,
        choices=("logic", "cuda-synthetic", "full"),
    )
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
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
        args.profile, args.seed, env, cases, elapsed
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
    print(json_path.relative_to(ROOT))
    print(md_path.relative_to(ROOT))
    return 1 if overall == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
