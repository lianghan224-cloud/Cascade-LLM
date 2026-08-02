#!/usr/bin/env python3
"""Export unified provider status without claiming untested compatibility."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming.hardware import default_provider_registry  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry = default_provider_registry()
    architectures = ("sm75", "sm80", "sm86", "sm89", "sm90")
    payload = {
        "schema_version": 1,
        "warning": "declared or compiled is not numerical/performance qualification",
        "architectures": {
            architecture: [
                {
                    "provider": item.provider_name,
                    "backend": item.backend_name,
                    "status": item.qualification_status,
                }
                for item in registry.list_providers()
                if architecture in item.supported_architectures
            ]
            for architecture in architectures
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
