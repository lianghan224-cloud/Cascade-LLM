#!/usr/bin/env python3
"""Prepare or build architecture-specific provider artifacts."""

import argparse
import subprocess
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming.hardware import ProviderBuildMetadata  # noqa: E402


KNOWN_ARCHITECTURES = ("sm75", "sm80", "sm86", "sm89", "sm90")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--architecture", choices=KNOWN_ARCHITECTURES)
    group.add_argument("--architectures")
    parser.add_argument("--provider", choices=("w8a16",), required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args, extra = parser.parse_known_args()
    architectures = (
        (args.architecture,)
        if args.architecture
        else tuple(item.strip() for item in args.architectures.split(",") if item.strip())
    )
    unknown = set(architectures).difference(KNOWN_ARCHITECTURES)
    if unknown:
        raise SystemExit("unknown architecture(s): {}".format(", ".join(sorted(unknown))))
    output = args.output or (
        PROJECT_ROOT / "build/provider_metadata/{}_{}.json".format(
            args.provider, "_".join(architectures)
        )
    )
    if args.metadata_only:
        metadata = ProviderBuildMetadata(
            provider="cutlass_{}".format(args.provider),
            provider_version="unverified",
            abi=0,
            compiled_architectures=(),
            weight_formats=("int8_per_channel",),
            activation_dtypes=("bf16", "fp16"),
            build_environment={
                "cuda": "unverified",
                "compiler": "unverified",
                "cutlass": "unverified",
            },
            status="declared",
        )
        metadata.write(output)
        print(output)
        return 0
    if architectures != ("sm86",):
        raise SystemExit(
            "no kernel implementation exists for {}; use --metadata-only to "
            "prepare an unqualified build declaration".format(",".join(architectures))
        )
    command = [
        sys.executable,
        "-m",
        "layer_streaming.providers.cutlass.build",
    ] + extra
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
