#!/usr/bin/env python3
"""Create a qualification checklist; this tool never promotes status itself."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming.hardware import HardwareDetector  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    hardware, runtime = HardwareDetector().detect(args.device)
    payload = {
        "schema_version": 1,
        "hardware": hardware.as_dict(),
        "runtime": runtime.as_dict(),
        "qualification_status": "unqualified",
        "automatic_promotion": False,
        "levels": {
            "level_1_environment": "observed",
            "level_2_base_backends": "not_run",
            "level_3_fused_operator": "not_run",
            "level_4_tiny_model": "not_run",
            "level_5_real_model": "not_run",
            "level_6_stability_performance": "not_run",
        },
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
