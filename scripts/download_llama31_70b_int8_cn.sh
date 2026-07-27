#!/usr/bin/env bash
set -Eeuo pipefail

# Download the public Llama 3.1 70B Instruct W8A8 checkpoint through a
# domestic Hugging Face mirror. Re-running the same command resumes downloads.

MODEL_ID="${MODEL_ID:-RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w8a8}"
MODEL_REVISION="${MODEL_REVISION:-8d0dcbba33eeef589b0a607e46abe05a5a6431a8}"
MODEL_DIR="${MODEL_DIR:-/ssd/cascade-llm/models/Llama-3.1-70B-Instruct-W8A8}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/ssd/cascade-llm}"
DOWNLOADER_VENV="${DOWNLOADER_VENV:-${DOWNLOAD_ROOT}/venvs/hf-download}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-4}"
MIN_FREE_GIB="${MIN_FREE_GIB:-85}"
PYPI_INDEX="${PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}"

unset http_proxy https_proxy all_proxy ftp_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY
export HF_ENDPOINT="${CASCADE_HF_ENDPOINT:-https://hf-mirror.com}"
export NO_PROXY="localhost,127.0.0.1,::1,hf-mirror.com,.hf-mirror.com"
export no_proxy="${NO_PROXY}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HOME="${HF_HOME:-${DOWNLOAD_ROOT}/hf-cache-70b-int8}"

probe_only=0
case "${1:-}" in
    "") ;;
    --probe) probe_only=1 ;;
    -h|--help)
        echo "Usage: bash scripts/download_llama31_70b_int8_cn.sh [--probe]"
        exit 0
        ;;
    *)
        echo "Unknown option: ${1}" >&2
        exit 2
        ;;
esac

if ! [[ "${DOWNLOAD_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DOWNLOAD_WORKERS must be a positive integer" >&2
    exit 2
fi

mkdir -p "${DOWNLOAD_ROOT}/venvs" "${HF_HOME}"
if [[ ! -x "${DOWNLOADER_VENV}/bin/python" ]]; then
    command -v uv >/dev/null 2>&1 || {
        echo "uv is required to create the downloader environment" >&2
        exit 1
    }
    uv venv --python 3.11 "${DOWNLOADER_VENV}"
fi

if ! "${DOWNLOADER_VENV}/bin/python" -c \
    'import huggingface_hub; assert tuple(map(int, huggingface_hub.__version__.split(".")[:2])) >= (0, 34)' \
    >/dev/null 2>&1; then
    uv pip install \
        --python "${DOWNLOADER_VENV}/bin/python" \
        --index-url "${PYPI_INDEX}" \
        "huggingface_hub==0.34.4"
fi

echo "Llama 3.1 70B Instruct W8A8 download"
echo "  model:       ${MODEL_ID}"
echo "  revision:    ${MODEL_REVISION}"
echo "  endpoint:    ${HF_ENDPOINT}"
echo "  destination: ${MODEL_DIR}"
echo "  workers:     ${DOWNLOAD_WORKERS}"
echo "  proxy vars:  cleared"

MODEL_ID="${MODEL_ID}" \
MODEL_REVISION="${MODEL_REVISION}" \
MODEL_DIR="${MODEL_DIR}" \
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS}" \
MIN_FREE_GIB="${MIN_FREE_GIB}" \
PROBE_ONLY="${probe_only}" \
"${DOWNLOADER_VENV}/bin/python" - <<'PY'
import json
import os
import shutil
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

model_id = os.environ["MODEL_ID"]
revision = os.environ["MODEL_REVISION"]
model_dir = Path(os.environ["MODEL_DIR"])
workers = int(os.environ["DOWNLOAD_WORKERS"])
minimum_free = int(os.environ["MIN_FREE_GIB"]) * 1024**3

info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
print(f"  resolved:    {info.sha}")
if info.sha != revision:
    raise SystemExit(f"revision mismatch: {info.sha} != {revision}")
if os.environ["PROBE_ONLY"] == "1":
    print("Mirror probe succeeded; no model data was downloaded.")
    raise SystemExit(0)

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
index = json.loads(
    (downloaded / "model.safetensors.index.json").read_text(
        encoding="utf-8"
    )
)
shards = sorted(set(index["weight_map"].values()))
missing = [
    name for name in shards
    if not (downloaded / name).is_file()
    or (downloaded / name).stat().st_size == 0
]
if missing:
    raise SystemExit(f"Missing or empty shards: {missing}")

dtypes = Counter()
tensor_count = 0
for name in shards:
    with (downloaded / name).open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    for tensor_name, metadata in header.items():
        if tensor_name != "__metadata__":
            dtypes[metadata["dtype"]] += 1
            tensor_count += 1

receipt = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_id": model_id,
    "resolved_revision": info.sha,
    "local_dir": str(downloaded),
    "weight_shards": len(shards),
    "tensor_count": tensor_count,
    "tensor_dtype_counts": dict(sorted(dtypes.items())),
    "endpoint": os.environ.get("HF_ENDPOINT"),
}
(downloaded / "cascade_download_receipt.json").write_text(
    json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(receipt, indent=2, ensure_ascii=False))
PY
