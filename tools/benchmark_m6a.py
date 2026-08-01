#!/usr/bin/env python3
"""Run and merge the frozen real-checkpoint M6A benchmark matrix."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    BENCHMARK_REPORT_SCHEMA_VERSION,
    BenchmarkSuiteReport,
)


def parse_lengths(value):
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("lengths must be positive integers")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefill-lengths", type=parse_lengths, default=(8, 128, 512))
    parser.add_argument("--decode-lengths", type=parse_lengths, default=(32, 128))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--vocab-chunk-mib", type=float, default=4.0)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ignore-memlock-limit", action="store_true")
    return parser.parse_args()


def load_part(path):
    return BenchmarkSuiteReport.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("warmup must be non-negative and repeats positive")
    part_dir = args.output.parent / (args.output.stem + "_parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    started = time.time()
    for prefill in args.prefill_lengths:
        for decode in args.decode_lengths:
            part = part_dir / "prefill_{}_decode_{}.json".format(prefill, decode)
            if not (args.resume and part.is_file()):
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "benchmark.py"),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--output",
                    str(part),
                    "--preset",
                    "m6a",
                    "--backend",
                    "checkpoint",
                    "--prefill-tokens",
                    str(prefill),
                    "--decode-tokens",
                    str(decode),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--slots",
                    str(args.slots),
                    "--prefetch-depth",
                    str(args.prefetch_depth),
                    "--vocab-chunk-mib",
                    str(args.vocab_chunk_mib),
                    "--embedding-placement",
                    "streamed",
                    "--lm-head-placement",
                    "streamed",
                    "--gpu-resident-weight-budget",
                    "0",
                    "--device",
                    args.device,
                ]
                if args.ignore_memlock_limit:
                    command.append("--ignore-memlock-limit")
                if args.baseline is not None:
                    command.extend(("--baseline", str(args.baseline)))
                print(
                    "running prefill={} decode={} ...".format(prefill, decode),
                    flush=True,
                )
                subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
            parts.append(load_part(part))

    results = tuple(item for report in parts for item in report.results)
    comparisons = tuple(
        item for report in parts for item in report.baseline_comparisons
    )
    rankings = []
    for prefill in args.prefill_lengths:
        for decode in args.decode_lengths:
            candidates = [
                item
                for item in results
                if item.status == "ok"
                and item.median is not None
                and item.case.prefill_tokens == prefill
                and item.case.decode_tokens == decode
            ]
            candidates.sort(key=lambda item: item.median.decode_ms_per_token)
            rankings.append(
                {
                    "prefill_tokens": prefill,
                    "decode_tokens": decode,
                    "granularity_order": [
                        item.case.granularity for item in candidates
                    ],
                    "decode_ms_per_token": [
                        item.median.decode_ms_per_token for item in candidates
                    ],
                }
            )
    stable = [
        bool(item.variability.get("stable", False))
        for item in results
        if item.status == "ok"
    ]
    first = parts[0]
    report = BenchmarkSuiteReport(
        schema_version=BENCHMARK_REPORT_SCHEMA_VERSION,
        synthetic_only=all(item.synthetic_only for item in parts),
        hardware=first.hardware,
        software={
            **first.software,
            "duration_seconds": time.time() - started,
            "preset": "m6a_frozen_real_baseline",
            "prefill_lengths": list(args.prefill_lengths),
            "decode_lengths": list(args.decode_lengths),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "granularities": ["matrix", "matrix_group", "layer"],
            "m6a_rankings": rankings,
            "stability_acceptance": {
                "threshold": 0.05,
                "passed_cases": sum(stable),
                "total_cases": len(stable),
                "all_passed": bool(stable) and all(stable),
            },
        },
        results=results,
        baseline_comparisons=comparisons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.to_json() + "\n", encoding="utf-8")
    print("wrote {} cases to {}".format(len(results), args.output))


if __name__ == "__main__":
    main()
