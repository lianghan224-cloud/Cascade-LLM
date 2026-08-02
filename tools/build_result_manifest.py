#!/usr/bin/env python3
"""Create a content manifest for result artifacts without embedding payloads."""

import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("real_results"))
    parser.add_argument(
        "--output", type=Path, default=Path("real_results/manifest.json")
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_payload(path, output):
    if path.resolve() == output.resolve():
        return False
    if path.name.endswith("summary.json"):
        return False
    if path.name in {"manifest.json", "compatibility_matrix.json"}:
        return False
    return path.suffix.lower() in {".json", ".pt", ".pth", ".npy", ".npz"}


def main():
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and is_payload(path, output):
            artifacts.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "git_policy": "ci_or_release_artifact",
                }
            )
    report = {
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(item["bytes"] for item in artifacts),
        "policy": (
            "payloads are untracked; Git stores summaries and this content "
            "manifest only"
        ),
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
