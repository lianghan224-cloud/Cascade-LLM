#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import platform
import subprocess

from layer_streaming.container_contracts import (
    ImageManifest,
    ProviderBundleManifest,
    load_versions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/opt/cascade"))
    parser.add_argument("--bundle", default="generic")
    parser.add_argument("--image-type", default="runtime")
    parser.add_argument("--git-commit", default="unverified")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    versions = load_versions(args.root / "docker/versions.env")
    bundle = ProviderBundleManifest.read(
        args.root / "docker/provider-bundles" / (args.bundle + ".json")
    )
    output = args.output or args.root / "image-manifest.json"
    manifest = ImageManifest(
        cascade_version=versions["CASCADE_VERSION"],
        git_commit=args.git_commit,
        execution_plan_schema=int(versions["EXECUTION_PLAN_SCHEMA"]),
        run_report_schema=int(versions["RUN_REPORT_SCHEMA"]),
        hardware_compatibility_schema=int(
            versions["HARDWARE_COMPATIBILITY_SCHEMA"]
        ),
        numerical_contract_version=int(
            versions["NUMERICAL_CONTRACT_VERSION"]
        ),
        python=versions["PYTHON_VERSION"],
        torch=versions["PYTORCH_VERSION"],
        cuda_runtime=versions["CUDA_VERSION"],
        provider_bundle=bundle.name,
        providers=tuple(provider.__dict__ for provider in bundle.providers),
        image_type=args.image_type,
    )
    manifest.write(output)
    print(output)


if __name__ == "__main__":
    main()
