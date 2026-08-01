#!/usr/bin/env python3
"""Inspect hardware once and emit a structured, non-qualification report."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming.hardware import (  # noqa: E402
    HardwareDetector,
    build_compatibility_report,
    default_provider_registry,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    hardware, runtime = HardwareDetector().detect(args.device)
    report = build_compatibility_report(
        hardware, runtime, default_provider_registry()
    )
    payload = report.to_json()
    if args.output is not None:
        report.write(args.output)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
