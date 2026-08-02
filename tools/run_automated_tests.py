#!/usr/bin/env python3
"""Run unittest discovery and emit a compact qualification summary."""

import argparse
import json
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "tests"
for path in (ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-dir", type=Path, default=TEST_ROOT)
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbosity", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    start_dir = args.start_dir.resolve()
    suite = unittest.defaultTestLoader.discover(
        str(start_dir), pattern=args.pattern
    )
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    duration = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "passed": result.wasSuccessful(),
        "tests_run": int(result.testsRun),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "duration_seconds": duration,
        "start_dir": (
            str(start_dir.relative_to(ROOT))
            if ROOT in start_dir.parents or start_dir == ROOT
            else str(start_dir)
        ),
        "pattern": args.pattern,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
