#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASCADE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${CASCADE_ROOT}/.env.local" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${CASCADE_ROOT}/.env.local"
    set +a
fi

# This downloader is intended for a direct, proxy-free terminal. Clear every
# common proxy spelling before the first metadata or file request.
unset \
    http_proxy https_proxy ftp_proxy all_proxy no_proxy \
    HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_DISABLE_PROGRESS_BARS

DEFAULT_REPO="meta-llama/CodeLlama-34b-hf"
MODEL_REPO="${CASCADE_MODEL_REPO:-${DEFAULT_REPO}}"
MODEL_REVISION="${CASCADE_MODEL_REVISION:-main}"
MODEL_DIR="${CASCADE_CODELLAMA34B_DIR:-/disk4/llm_model/CodeLlama-34b-hf}"
MIRROR_ENDPOINT="${CASCADE_HF_ENDPOINT:-https://hf-mirror.com}"
DOWNLOAD_WORKERS="${CASCADE_DOWNLOAD_WORKERS:-4}"
DOWNLOAD_LOG="${CASCADE_DOWNLOAD_LOG:-}"
MODE="download"

usage() {
    cat <<'EOF'
Usage: scripts/download_codellama34b_bf16.sh [options]

Download the official meta-llama/CodeLlama-34b-hf BF16 safetensors through a
Hugging Face mirror. The script clears inherited proxy variables, supports
resume, prints progress in real time, and records the same output in a log.

Options:
  --repo REPO          Hugging Face repository id
  --revision REV       Branch, tag, or immutable commit (default: main)
  --output DIR         Checkpoint destination
  --workers N          Concurrent download workers (default: 4)
  --log-file FILE      Persistent download log path
  --dry-run            Check metadata/disk space and print the command only
  --verify-only        Validate an existing checkpoint without network access
  -h, --help           Show this help

Environment overrides:
  CASCADE_MODEL_REPO
  CASCADE_MODEL_REVISION
  CASCADE_CODELLAMA34B_DIR
  CASCADE_HF_ENDPOINT
  CASCADE_HF_HOME
  CASCADE_DOWNLOAD_WORKERS
  CASCADE_DOWNLOAD_LOG
  HF_TOKEN             Token for an account that has accepted the Meta license

Defaults:
  mirror:              https://hf-mirror.com
  output:              /disk4/llm_model/CodeLlama-34b-hf
  cache:               <output>.hf-cache
  log:                 <output>.download.log
EOF
}

while (($#)); do
    case "$1" in
        --repo)
            [[ $# -ge 2 ]] || {
                printf 'error: --repo requires a value\n' >&2
                exit 2
            }
            MODEL_REPO="$2"
            shift 2
            ;;
        --revision)
            [[ $# -ge 2 ]] || {
                printf 'error: --revision requires a value\n' >&2
                exit 2
            }
            MODEL_REVISION="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || {
                printf 'error: --output requires a value\n' >&2
                exit 2
            }
            MODEL_DIR="$2"
            shift 2
            ;;
        --workers)
            [[ $# -ge 2 ]] || {
                printf 'error: --workers requires a value\n' >&2
                exit 2
            }
            DOWNLOAD_WORKERS="$2"
            shift 2
            ;;
        --log-file)
            [[ $# -ge 2 ]] || {
                printf 'error: --log-file requires a value\n' >&2
                exit 2
            }
            DOWNLOAD_LOG="$2"
            shift 2
            ;;
        --dry-run)
            MODE="dry-run"
            shift
            ;;
        --verify-only)
            MODE="verify-only"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'error: unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODEL_REPO}" || -z "${MODEL_REVISION}" ]]; then
    printf 'error: repository and revision cannot be empty\n' >&2
    exit 2
fi
if [[ ! "${DOWNLOAD_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: workers must be a positive integer\n' >&2
    exit 2
fi

MODEL_DIR="$(readlink -m -- "${MODEL_DIR}")"
if [[ "${MODEL_DIR}" == "/" ]]; then
    printf 'error: refusing to use / as the model directory\n' >&2
    exit 2
fi
if [[ -z "${DOWNLOAD_LOG}" ]]; then
    DOWNLOAD_LOG="${MODEL_DIR}.download.log"
fi
DOWNLOAD_LOG="$(readlink -m -- "${DOWNLOAD_LOG}")"
if [[ "${DOWNLOAD_LOG}" == "/" ]]; then
    printf 'error: refusing to use / as the log path\n' >&2
    exit 2
fi

PYTHON="${CASCADE_ROOT}/.venv/bin/python"
HF_CLI="${CASCADE_ROOT}/.venv/bin/huggingface-cli"
if [[ ! -x "${PYTHON}" || ! -x "${HF_CLI}" ]]; then
    printf 'error: project virtualenv is incomplete; expected %s and %s\n' \
        "${PYTHON}" "${HF_CLI}" >&2
    exit 2
fi

export HF_ENDPOINT="${MIRROR_ENDPOINT}"
export HF_HOME="${CASCADE_HF_HOME:-${MODEL_DIR}.hf-cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "${MODEL_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" \
    "$(dirname -- "${DOWNLOAD_LOG}")"

# Send one live stream to both the terminal and a persistent logfile. The
# unbuffered Python environment keeps metadata/errors visible immediately;
# huggingface-cli's per-file progress bars are forwarded as they are emitted.
exec > >(tee -a "${DOWNLOAD_LOG}") 2>&1

on_exit() {
    local status=$?
    if ((status == 0)); then
        printf '[%s] downloader finished successfully\n' "$(date '+%F %T%z')"
    else
        printf '[%s] downloader failed with exit code %s\n' \
            "$(date '+%F %T%z')" "${status}"
    fi
}
trap on_exit EXIT

verify_checkpoint() {
    "${PYTHON}" - "${CASCADE_ROOT}" "${MODEL_DIR}" <<'PY'
from pathlib import Path
import json
import struct
import sys

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
sys.path.insert(0, str(root))

from transformers import AutoConfig, AutoTokenizer

from layer_streaming import (
    CheckpointManifest,
    ExecutionPolicy,
    adapter_for_config,
)

index_path = checkpoint / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit("missing model.safetensors.index.json")

index = json.loads(index_path.read_text(encoding="utf-8"))
expected_shards = sorted(set(index.get("weight_map", {}).values()))
if not expected_shards:
    raise SystemExit("safetensors index contains no weight shards")

missing = [name for name in expected_shards if not (checkpoint / name).is_file()]
empty = [
    name
    for name in expected_shards
    if (checkpoint / name).is_file() and (checkpoint / name).stat().st_size == 0
]
if missing or empty:
    raise SystemExit(
        "checkpoint shards are incomplete; missing={}, empty={}".format(
            missing, empty
        )
    )

dtypes = set()
tensor_count = 0
for name in expected_shards:
    shard = checkpoint / name
    with shard.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise SystemExit("invalid safetensors header: {}".format(shard))
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size))
    for tensor_name, tensor in header.items():
        if tensor_name == "__metadata__":
            continue
        tensor_count += 1
        dtypes.add(tensor["dtype"])

if dtypes != {"BF16"}:
    raise SystemExit(
        "expected an all-BF16 checkpoint, found dtypes: {}".format(
            ", ".join(sorted(dtypes)) or "none"
        )
    )

config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
adapter = adapter_for_config(config)
geometry = adapter.build_geometry(config)
policy = ExecutionPolicy.from_config(config)
specs = adapter.enumerate_weights(config, policy=policy)
manifest = CheckpointManifest.from_path(checkpoint)
validation = manifest.validate(specs)
if not validation.ok:
    raise SystemExit(validation.format_errors())

print("checkpoint validation: ok")
print("checkpoint dtype:      BF16")
print("safetensor shards:     {}".format(len(expected_shards)))
print("tensor entries:        {}".format(tensor_count))
print("model_type:            {}".format(geometry.model_type))
print("layers:                {}".format(geometry.num_hidden_layers))
print("hidden_size:           {}".format(geometry.hidden_size))
print("attention_heads:       {}".format(geometry.num_attention_heads))
print("kv_heads:              {}".format(geometry.num_key_value_heads))
print("vocab_size:            {}".format(geometry.vocab_size))
print("checkpoint_bytes:      {}".format(manifest.total_bytes))
print("tokenizer:             {}".format(type(tokenizer).__name__))
PY
}

printf '[%s] CodeLlama BF16 downloader started\n' "$(date '+%F %T%z')"
printf 'Proxy environment: cleared\n'
printf 'Mirror endpoint:   %s\n' "${HF_ENDPOINT}"
printf 'Repository:        %s\n' "${MODEL_REPO}"
printf 'Requested revision:%s\n' " ${MODEL_REVISION}"
printf 'Destination:       %s\n' "${MODEL_DIR}"
printf 'Persistent log:    %s\n' "${DOWNLOAD_LOG}"
printf 'Download workers:  %s\n' "${DOWNLOAD_WORKERS}"

if [[ "${MODE}" == "verify-only" ]]; then
    verify_checkpoint
    exit 0
fi

read -r RESOLVED_REVISION REMOTE_BYTES REMOTE_FILES < <(
    "${PYTHON}" - "${MODEL_REPO}" "${MODEL_REVISION}" <<'PY'
import fnmatch
import os
import sys

from huggingface_hub import HfApi

repo_id, revision = sys.argv[1:]
patterns = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "LICENSE*",
    "USE_POLICY*",
    "README*",
)
token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
try:
    info = HfApi(
        endpoint=os.environ["HF_ENDPOINT"],
        token=token,
    ).model_info(repo_id, revision=revision, files_metadata=True)
except Exception as error:
    print(
        "mirror metadata request failed for {}@{}: {}".format(
            repo_id, revision, error
        ),
        file=sys.stderr,
    )
    print(
        "Accept the Meta Code Llama license on huggingface.co, then export "
        "HF_TOKEN in this terminal before rerunning.",
        file=sys.stderr,
    )
    raise SystemExit(2)

selected = [
    item
    for item in info.siblings
    if "/" not in item.rfilename
    and any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns)
]
names = {item.rfilename for item in selected}
shards = sorted(name for name in names if name.endswith(".safetensors"))
if "model.safetensors.index.json" not in names or len(shards) != 7:
    raise SystemExit(
        "expected the CodeLlama-34B safetensors index and 7 shards; found "
        "index={}, shards={}".format(
            "model.safetensors.index.json" in names, len(shards)
        )
    )

total = sum(int(item.size or 0) for item in selected)
if total <= 0:
    raise SystemExit("mirror returned checkpoint files without size metadata")
print(info.sha, total, len(selected))
PY
)

EXISTING_BYTES="$(
    "${PYTHON}" - "${MODEL_DIR}" <<'PY'
import fnmatch
from pathlib import Path
import sys

root = Path(sys.argv[1])
patterns = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "LICENSE*",
    "USE_POLICY*",
    "README*",
)
total = sum(
    path.stat().st_size
    for path in root.iterdir()
    if path.is_file()
    and any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
)
print(total)
PY
)"
AVAILABLE_BYTES="$(
    df -Pk "${MODEL_DIR}" | awk 'NR == 2 {printf "%.0f", $4 * 1024}'
)"
if ((REMOTE_BYTES > EXISTING_BYTES)); then
    MISSING_BYTES=$((REMOTE_BYTES - EXISTING_BYTES))
else
    MISSING_BYTES=0
fi
SAFETY_BYTES=$((2 * 1024 * 1024 * 1024))
REQUIRED_BYTES=$((MISSING_BYTES + SAFETY_BYTES))
if ((AVAILABLE_BYTES < REQUIRED_BYTES)); then
    "${PYTHON}" - "${AVAILABLE_BYTES}" "${REQUIRED_BYTES}" <<'PY'
import sys

available, required = (int(item) for item in sys.argv[1:])
gib = 1024 ** 3
raise SystemExit(
    "insufficient disk space: available {:.2f} GiB, require {:.2f} GiB".format(
        available / gib, required / gib
    )
)
PY
fi

DOWNLOAD_COMMAND=(
    "${HF_CLI}" download "${MODEL_REPO}"
    --revision "${RESOLVED_REVISION}"
    --local-dir "${MODEL_DIR}"
    --include
        "*.json"
        "*.safetensors"
        "*.model"
        "*.txt"
        "LICENSE*"
        "USE_POLICY*"
        "README*"
    --exclude
        "*.bin"
        "*.pth"
        "*.pt"
        "*.gguf"
        "original/*"
    --max-workers "${DOWNLOAD_WORKERS}"
)

printf 'Resolved revision: %s\n' "${RESOLVED_REVISION}"
printf 'Selected files:    %s\n' "${REMOTE_FILES}"
"${PYTHON}" - "${REMOTE_BYTES}" "${MISSING_BYTES}" "${AVAILABLE_BYTES}" <<'PY'
import sys

remote, missing, available = (int(item) for item in sys.argv[1:])
gib = 1024 ** 3
print("Remote size:       {:.2f} GiB".format(remote / gib))
print("Estimated missing: {:.2f} GiB".format(missing / gib))
print("Disk available:    {:.2f} GiB".format(available / gib))
PY

if [[ "${MODE}" == "dry-run" ]]; then
    printf 'Download command:  '
    printf '%q ' "${DOWNLOAD_COMMAND[@]}"
    printf '\n'
    exit 0
fi

LOCK_PATH="${MODEL_DIR}.download.lock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    printf 'error: another download holds %s\n' "${LOCK_PATH}" >&2
    exit 2
fi

printf '[%s] starting/resuming file transfer\n' "$(date '+%F %T%z')"
"${DOWNLOAD_COMMAND[@]}"

printf '[%s] transfer finished; validating shards and BF16 dtype\n' \
    "$(date '+%F %T%z')"
verify_checkpoint

MARKER_PATH="${MODEL_DIR}.cascade-download.json"
"${PYTHON}" - "${MARKER_PATH}" "${MODEL_REPO}" \
    "${RESOLVED_REVISION}" "${HF_ENDPOINT}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "repository": sys.argv[2],
    "revision": sys.argv[3],
    "endpoint": sys.argv[4],
    "weight_dtype": "BF16",
    "weight_format": "safetensors",
}
path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

printf 'Download and BF16 validation completed: %s\n' "${MODEL_DIR}"
