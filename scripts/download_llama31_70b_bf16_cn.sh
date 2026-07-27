#!/usr/bin/env bash
set -Eeuo pipefail

# AirLLM compression=8bit must start from an ordinary BF16 checkpoint. This
# public mirror contains only the Transformers-format copy (about 141 GB).
MODEL_ID="${MODEL_ID:-unsloth/Meta-Llama-3.1-70B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-1fdd0a465a29664d155aee8a9f77c55a65cc8d5f}"
MODEL_DIR="${MODEL_DIR:-/disk2/home/guest/lianghan/models/Llama-3.1-70B-Instruct-BF16}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/ssd/cascade-llm}"
DOWNLOADER_VENV="${DOWNLOADER_VENV:-${DOWNLOAD_ROOT}/venvs/hf-download}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-4}"
MIN_FREE_GIB="${MIN_FREE_GIB:-155}"

unset http_proxy https_proxy all_proxy ftp_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY
export HF_ENDPOINT="${CASCADE_HF_ENDPOINT:-https://hf-mirror.com}"
export NO_PROXY="localhost,127.0.0.1,::1,hf-mirror.com,.hf-mirror.com"
export no_proxy="${NO_PROXY}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HOME="${HF_HOME:-${DOWNLOAD_ROOT}/hf-cache-70b-bf16}"

if [[ ! -x "${DOWNLOADER_VENV}/bin/python" ]]; then
    echo "Missing downloader environment: ${DOWNLOADER_VENV}" >&2
    exit 1
fi

MODEL_ID="${MODEL_ID}" \
MODEL_REVISION="${MODEL_REVISION}" \
MODEL_DIR="${MODEL_DIR}" \
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS}" \
MIN_FREE_GIB="${MIN_FREE_GIB}" \
"${DOWNLOADER_VENV}/bin/python" - <<'PY'
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

model_id = os.environ["MODEL_ID"]
revision = os.environ["MODEL_REVISION"]
model_dir = Path(os.environ["MODEL_DIR"])
workers = int(os.environ["DOWNLOAD_WORKERS"])
minimum_free = int(os.environ["MIN_FREE_GIB"]) * 1024**3

info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
remote_bytes = sum(sibling.size or 0 for sibling in info.siblings)
print(f"model:       {model_id}", flush=True)
print(f"revision:    {info.sha}", flush=True)
print(f"endpoint:    {os.environ.get('HF_ENDPOINT')}", flush=True)
print(f"repository:  {remote_bytes / 1e9:.3f} GB", flush=True)
print(f"destination: {model_dir}", flush=True)
if info.sha != revision:
    raise SystemExit(f"revision mismatch: {info.sha} != {revision}")

model_dir.mkdir(parents=True, exist_ok=True)
existing = sum(
    path.stat().st_size for path in model_dir.rglob("*") if path.is_file()
)
capacity = shutil.disk_usage(model_dir).free + existing
if capacity < minimum_free:
    raise SystemExit(
        f"Only {capacity / 1024**3:.1f} GiB free-plus-resumable capacity; "
        f"{minimum_free / 1024**3:.0f} GiB required"
    )

downloaded = Path(
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=model_dir,
        max_workers=workers,
    )
)
index_path = downloaded / "model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(set(index["weight_map"].values()))
missing = [
    name for name in shards
    if not (downloaded / name).is_file()
    or (downloaded / name).stat().st_size == 0
]
if missing:
    raise SystemExit(f"Missing or empty shards: {missing}")

receipt = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_id": model_id,
    "resolved_revision": info.sha,
    "local_dir": str(downloaded),
    "weight_shards": len(shards),
    "repository_bytes": remote_bytes,
    "endpoint": os.environ.get("HF_ENDPOINT"),
}
(downloaded / "cascade_download_receipt.json").write_text(
    json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(receipt, indent=2, ensure_ascii=False))
PY
