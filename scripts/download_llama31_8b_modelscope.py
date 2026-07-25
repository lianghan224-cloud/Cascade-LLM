#!/usr/bin/env python3
"""Download a hash-pinned BF16 Llama-3.1-8B mirror from ModelScope."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests


OFFICIAL_MODEL_ID = "meta-llama/Llama-3.1-8B"
MIRROR_REPO_ID = "AI-ModelScope/Meta-Llama-3.1-8B"
MIRROR_REVISION = "d61242bf99a3b4d26dea55101e1115627a07c4a9"
FILES = [
    ("LICENSE", 7627, "64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369"),
    ("USE_POLICY.md", 4691, "a568f2ebc73cec3fd74ba2afd992d4e945a8c7a9d851f9b66163aac834b7b859"),
    ("config.json", 826, "54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd"),
    ("configuration.json", 73, "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9"),
    ("generation_config.json", 185, "e645194d2dd27c86ed34a5a23f306c1a0fd79a42123c26551b7251c700a3379d"),
    (
        "model-00001-of-00004.safetensors",
        4976698672,
        "f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b",
    ),
    (
        "model-00002-of-00004.safetensors",
        4999802720,
        "c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0",
    ),
    (
        "model-00003-of-00004.safetensors",
        4915916176,
        "d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383",
    ),
    (
        "model-00004-of-00004.safetensors",
        1168138808,
        "e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d",
    ),
    (
        "model.safetensors.index.json",
        23950,
        "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
    ),
    (
        "special_tokens_map.json",
        73,
        "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
    ),
    (
        "tokenizer.json",
        9085658,
        "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa",
    ),
    (
        "tokenizer_config.json",
        50500,
        "8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a",
    ),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(path):
    repo = quote(MIRROR_REPO_ID, safe="/")
    file_path = quote(path, safe="/")
    return (
        "https://modelscope.cn/api/v1/models/{}/repo"
        "?Revision={}&FilePath={}".format(
            repo,
            MIRROR_REVISION,
            file_path,
        )
    )


def download_one(session, target, expected_size, expected_sha256):
    if target.exists():
        actual_size = target.stat().st_size
        if actual_size == expected_size and sha256(target) == expected_sha256:
            print("[verified] {}".format(target.name), flush=True)
            return
        raise RuntimeError(
            "{} exists but does not match the pinned manifest".format(target)
        )

    partial = target.with_name(target.name + ".partial")
    for attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": "bytes={}-".format(offset)} if offset else {}
        try:
            with session.get(
                download_url(target.name),
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if offset > 0 and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                downloaded = offset
                next_report = downloaded + 256 * 1024 * 1024
                with partial.open(mode) as output:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(
                                "[download] {} {:.2f}/{:.2f} GiB".format(
                                    target.name,
                                    downloaded / 1024 ** 3,
                                    expected_size / 1024 ** 3,
                                ),
                                flush=True,
                            )
                            next_report += 256 * 1024 * 1024
            if partial.stat().st_size != expected_size:
                raise RuntimeError(
                    "size {} expected {}".format(
                        partial.stat().st_size,
                        expected_size,
                    )
                )
            actual_sha256 = sha256(partial)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "sha256 {} expected {}".format(
                        actual_sha256,
                        expected_sha256,
                    )
                )
            partial.replace(target)
            print("[verified] {}".format(target.name), flush=True)
            return
        except (OSError, requests.RequestException, RuntimeError) as error:
            if attempt == 6:
                raise
            delay = min(30, 2 ** attempt)
            print(
                "[retry {}/6] {}: {}; waiting {}s".format(
                    attempt,
                    target.name,
                    error,
                    delay,
                ),
                flush=True,
            )
            time.sleep(delay)


def parse_args():
    default_model = os.environ.get(
        "CASCADE_LLAMA31_8B",
        "/ssd/cascade-llm/models/Llama-3.1-8B",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, default=Path(default_model))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("real_results/llama31_8b_checkpoint.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers["User-Agent"] = "Cascade-LLM/real-experiment"
        for name, size, digest in FILES:
            download_one(session, args.local_dir / name, size, digest)

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_model_id": OFFICIAL_MODEL_ID,
        "source": "ModelScope verified mirror",
        "source_repo_id": MIRROR_REPO_ID,
        "resolved_revision": MIRROR_REVISION,
        "local_dir": str(args.local_dir),
        "files": [
            {"name": name, "bytes": size, "sha256": digest}
            for name, size, digest in FILES
        ],
        "total_bytes": sum(size for _, size, _ in FILES),
        "safetensor_bytes": sum(
            size for name, size, _ in FILES if name.endswith(".safetensors")
        ),
        "note": (
            "The mirror exposes the same root BF16 safetensor sizes as the "
            "gated Hugging Face repository. Every downloaded byte is checked "
            "against this revision-pinned ModelScope manifest."
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
