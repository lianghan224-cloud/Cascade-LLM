#!/usr/bin/env python3
"""Download an authorized immutable Llama-3.1-8B snapshot and write a receipt."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "meta-llama/Llama-3.1-8B"
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    default_model = os.environ.get(
        "CASCADE_LLAMA31_8B",
        "/ssd/cascade-llm/models/Llama-3.1-8B",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", type=Path, default=Path(default_model))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("real_results/llama31_8b_checkpoint.json"),
    )
    return parser.parse_args()


def main():
    from huggingface_hub import HfApi, HfFolder, snapshot_download

    args = parse_args()
    if not HfFolder.get_token():
        raise SystemExit(
            "No Hugging Face token found. Accept the Meta license and run "
            "`huggingface-cli login` first."
        )
    api = HfApi()
    info = api.model_info(args.repo_id, revision=args.revision)
    immutable_revision = info.sha
    args.local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=immutable_revision,
            local_dir=args.local_dir,
            allow_patterns=ALLOW_PATTERNS,
            resume_download=True,
        )
    )

    files = []
    for path in sorted(downloaded.iterdir()):
        if not path.is_file():
            continue
        item = {
            "name": path.name,
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".safetensors" or path.name.endswith(".json"):
            item["sha256"] = sha256(path)
        files.append(item)

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": immutable_revision,
        "local_dir": str(args.local_dir),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "safetensor_bytes": sum(
            item["bytes"]
            for item in files
            if item["name"].endswith(".safetensors")
        ),
        "note": (
            "Token and checkpoint are intentionally excluded from Git; this "
            "receipt contains only revision, sizes, and content hashes."
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
