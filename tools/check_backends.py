#!/usr/bin/env python3
"""Report backend availability without silently selecting a fallback."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import backend_capabilities  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument(
        "--provider", choices=("none", "cutlass"), default="none"
    )
    parser.add_argument("--provider-library")
    args = parser.parse_args()
    if args.provider == "cutlass":
        from layer_streaming.providers.cutlass import (
            load_cutlass_w8a16_provider,
        )

        load_cutlass_w8a16_provider(args.provider_library)
    capabilities = backend_capabilities(args.device)
    payload = {
        "device": args.device,
        "backends": capabilities,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    missing = [
        name
        for name in args.require
        if name not in capabilities or not capabilities[name]["available"]
    ]
    if missing:
        print(
            "required backend(s) unavailable: {}".format(
                ", ".join(missing)
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
