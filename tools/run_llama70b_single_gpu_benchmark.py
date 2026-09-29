#!/usr/bin/env python3
"""Run the requested four-group real 70B single-GPU benchmark, resumably."""

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from tools import summarize_llama70b_single_gpu_benchmark as benchmark_summary
except ImportError:  # Direct execution places tools/ rather than repo root on sys.path.
    import summarize_llama70b_single_gpu_benchmark as benchmark_summary


ROOT = Path(__file__).resolve().parents[1]
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("baseline", "pipeline", "budget", "stability", "all"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "reports" / "llama70b_single_gpu_benchmark",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/tmp/cascade_llama70b_single_gpu_benchmark_raw"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--best-granularity", choices=("matrix", "layer"), default=None
    )
    parser.add_argument("--best-slots", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument(
        "--best-prefetch-depth", type=int, choices=(0, 1, 2), default=None
    )
    parser.add_argument(
        "--best-budget-gib", type=int, choices=(0, 4, 8, 12, 16, 20), default=None
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_dir():
        parser.error("checkpoint must be an existing directory")
    return args


def cases_for_stage(args, stage):
    base = {
        "granularity": "matrix",
        "slots": 2,
        "prefetch_depth": 2,
        "budget_gib": 0,
    }
    if stage == "baseline":
        return [
            dict(
                base,
                case_id="p{}_d{}".format(prompt, decode),
                prompt=prompt,
                decode=decode,
                repeat=1,
            )
            for prompt, decode in itertools.product(
                (512, 2048, 4096, 8192), (1, 8, 32)
            )
        ]
    if stage == "pipeline":
        result = []
        for granularity, slots, depth in itertools.product(
            ("matrix", "layer"), (1, 2, 3), (0, 1, 2)
        ):
            result.append(
                {
                    "case_id": "{}_s{}_p{}".format(
                        granularity, slots, depth
                    ),
                    "prompt": 2048,
                    "decode": 32,
                    "repeat": 1,
                    "granularity": granularity,
                    "slots": slots,
                    "prefetch_depth": depth,
                    "budget_gib": 0,
                    "executable": True,
                    "reason": None,
                }
            )
        return result
    selected = {
        "granularity": args.best_granularity,
        "slots": args.best_slots,
        "prefetch_depth": args.best_prefetch_depth,
    }
    if stage == "budget":
        return [
            dict(
                selected,
                case_id="budget_{}gib".format(budget),
                prompt=2048,
                decode=32,
                repeat=1,
                budget_gib=budget,
            )
            for budget in (0, 4, 8, 12, 16, 20)
        ]
    if stage == "stability":
        return [
            dict(
                selected,
                case_id="final_4096_d32_r5",
                prompt=4096,
                decode=32,
                repeat=5,
                budget_gib=args.best_budget_gib,
            )
        ]
    raise ValueError(stage)


def _bundle_valid(bundle):
    summary_path = bundle / "summary.json"
    cases_path = bundle / "cases.json"
    if not summary_path.is_file() or not cases_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = json.loads(cases_path.read_text(encoding="utf-8")).get("cases", [])
    if summary.get("status") != "SMOKE_ONLY" or not cases:
        return False
    for case in cases:
        monitor = case.get("gpu_activity_monitor") or {}
        if case.get("status") != "PASS" or monitor.get(
            "external_activity_observed"
        ):
            return False
        admission = case.get("gpu_admission") or {}
        if any(
            int((admission.get(boundary) or {}).get(
                "selected_gpu_external_compute_processes"
            ) or 0)
            for boundary in ("before", "after")
        ):
            return False
    return True


def _command(args, case, bundle):
    return [
        str(args.python),
        str(ROOT / "tools" / "qualify_llama70b_single_request.py"),
        "--mode",
        "smoke",
        "--require-uncontended-benchmark",
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        args.device,
        "--case",
        "{}:{}".format(case["prompt"], case["decode"]),
        "--repeat",
        str(case["repeat"]),
        "--weight-store",
        "pinned_staging",
        "--granularity",
        case["granularity"],
        "--slots",
        str(case["slots"]),
        "--prefetch-depth",
        str(case["prefetch_depth"]),
        "--embedding-placement",
        "resident",
        "--lm-head-placement",
        "streamed",
        "--kv-page-size",
        "16",
        "--kv-attention-backend",
        "generic_cuda",
        "--kv-prefill-backend",
        "gather_sdpa_prefill",
        "--gpu-resident-weight-budget",
        "{}GiB".format(case["budget_gib"]),
        "--ignore-memlock-limit",
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output-dir",
        str(bundle),
        "--raw-dir",
        str(args.raw_root),
        "--python",
        str(args.python),
    ]


def _write_selection(args, name, selection):
    path = args.output_root / "selected_config.json"
    existing = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema_version": 1}
    )
    existing[name] = selection
    path.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_best_config(args, stage):
    """Fill unspecified downstream parameters from clean prior-stage rows."""

    rows = benchmark_summary.collect(args.output_root)
    if stage == "budget":
        selection = benchmark_summary.select_pipeline_configuration(rows)
        if selection is not None:
            if args.best_granularity is None:
                args.best_granularity = selection["granularity"]
            if args.best_slots is None:
                args.best_slots = selection["slots"]
            if args.best_prefetch_depth is None:
                args.best_prefetch_depth = selection["prefetch_depth"]
            _write_selection(args, "pipeline", selection)
        missing = [
            name
            for name in (
                "best_granularity",
                "best_slots",
                "best_prefetch_depth",
            )
            if getattr(args, name) is None
        ]
        if missing:
            raise RuntimeError(
                "budget stage requires a complete clean pipeline matrix or "
                "explicit --best-granularity/--best-slots/"
                "--best-prefetch-depth"
            )
    elif stage == "stability":
        selection = benchmark_summary.select_budget_configuration(rows)
        if selection is not None:
            if args.best_granularity is None:
                args.best_granularity = selection["granularity"]
            if args.best_slots is None:
                args.best_slots = selection["slots"]
            if args.best_prefetch_depth is None:
                args.best_prefetch_depth = selection["prefetch_depth"]
            if args.best_budget_gib is None:
                args.best_budget_gib = int(
                    selection["gpu_resident_weight_budget_gib"]
                )
            _write_selection(args, "budget", selection)
        missing = [
            name
            for name in (
                "best_granularity",
                "best_slots",
                "best_prefetch_depth",
                "best_budget_gib",
            )
            if getattr(args, name) is None
        ]
        if missing:
            raise RuntimeError(
                "stability stage requires a complete clean budget matrix or "
                "all explicit --best-* parameters"
            )


def run_stage(args, stage):
    plan = cases_for_stage(args, stage)
    stage_root = args.output_root / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    (stage_root / "execution_plan.json").write_text(
        json.dumps({"schema_version": 1, "stage": stage, "cases": plan}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    for case in plan:
        bundle = stage_root / case["case_id"]
        if not case.get("executable", True):
            continue
        if not args.force and _bundle_valid(bundle):
            print("resume PASS {}".format(case["case_id"]), flush=True)
            continue
        bundle.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            _command(args, case, bundle),
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
        (bundle / "runner_stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (bundle / "runner_stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            print(
                "stage {} stopped at {} with exit {}".format(
                    stage, case["case_id"], completed.returncode
                ),
                file=sys.stderr,
            )
            return 2
        print("completed {}".format(case["case_id"]), flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    stages = (
        ("baseline", "pipeline", "budget", "stability")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        try:
            resolve_best_config(args, stage)
        except RuntimeError as error:
            print("cannot start {}: {}".format(stage, error), file=sys.stderr)
            return 2
        status = run_stage(args, stage)
        if status:
            return status
        if stage == "pipeline":
            selection = benchmark_summary.select_pipeline_configuration(
                benchmark_summary.collect(args.output_root)
            )
            if selection is not None:
                _write_selection(args, "pipeline", selection)
        elif stage == "budget":
            selection = benchmark_summary.select_budget_configuration(
                benchmark_summary.collect(args.output_root)
            )
            if selection is not None:
                _write_selection(args, "budget", selection)
    subprocess.run(
        [
            str(args.python),
            str(ROOT / "tools" / "summarize_llama70b_single_gpu_benchmark.py"),
            "--input-root",
            str(args.output_root),
            "--output-dir",
            str(args.output_root),
        ],
        cwd=str(ROOT),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
