#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASCADE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${CASCADE_ROOT}/.env.local" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${CASCADE_ROOT}/.env.local"
    set +a
fi

# Downloading must not inherit a terminal/VPN proxy.  Both lower-case and
# upper-case variants are cleared before the first network request.
unset \
    http_proxy https_proxy ftp_proxy all_proxy no_proxy \
    HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

DEFAULT_REPO="unsloth/Meta-Llama-3.1-8B-Instruct"
DEFAULT_REVISION="a2856192dd7c25b842431f39c179a6c2c2f627d1"
MODEL_REPO="${CASCADE_MODEL_REPO:-${DEFAULT_REPO}}"
MODEL_REVISION="${CASCADE_MODEL_REVISION:-}"
MODEL_DIR="${CASCADE_LLAMA31_8B:-/ssd/cascade-llm/models/Llama-3.1-8B-Instruct}"
MIRROR_ENDPOINT="${CASCADE_HF_ENDPOINT:-https://hf-mirror.com}"
DOWNLOAD_WORKERS="${CASCADE_DOWNLOAD_WORKERS:-4}"
MODE="download"

usage() {
    cat <<'EOF'
Usage: scripts/download_llama31_8b.sh [options]

Download a full-precision Llama 3.1 8B safetensors checkpoint through a
domestic Hugging Face mirror. Proxy environment variables are always removed
inside this script before any network access.

Options:
  --repo REPO          Hugging Face repository id
  --revision REV       Branch, tag, or immutable commit
  --output DIR         Checkpoint destination
  --workers N          Concurrent download workers (default: 4)
  --dry-run            Check mirror metadata/disk space and print the command
  --verify-only        Validate an existing local checkpoint without network
  -h, --help           Show this help

Environment overrides:
  CASCADE_MODEL_REPO
  CASCADE_MODEL_REVISION
  CASCADE_LLAMA31_8B
  CASCADE_HF_ENDPOINT
  CASCADE_DOWNLOAD_WORKERS
  HF_TOKEN             Optional; required only for a gated repository whose
                       license has already been accepted on huggingface.co
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

if [[ -z "${MODEL_REVISION}" ]]; then
    if [[ "${MODEL_REPO}" == "${DEFAULT_REPO}" ]]; then
        MODEL_REVISION="${DEFAULT_REVISION}"
    else
        MODEL_REVISION="main"
    fi
fi
if [[ ! "${DOWNLOAD_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: workers must be a positive integer\n' >&2
    exit 2
fi
if [[ -z "${MODEL_REPO}" || -z "${MODEL_REVISION}" ]]; then
    printf 'error: repository and revision cannot be empty\n' >&2
    exit 2
fi

MODEL_DIR="$(readlink -m -- "${MODEL_DIR}")"
if [[ "${MODEL_DIR}" == "/" ]]; then
    printf 'error: refusing to use / as the model directory\n' >&2
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
export HF_HOME="${HF_HOME:-/ssd/cascade-llm/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_DISABLE_TELEMETRY=1

verify_checkpoint() {
    "${PYTHON}" - "${CASCADE_ROOT}" "${MODEL_DIR}" <<'PY'
from pathlib import Path
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

if [[ "${MODE}" == "verify-only" ]]; then
    verify_checkpoint
    exit 0
fi

mkdir -p "${MODEL_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}"

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
        "For a gated Meta repository, accept its license on huggingface.co "
        "and export HF_TOKEN before rerunning.",
        file=sys.stderr,
    )
    raise SystemExit(2)

selected = [
    item
    for item in info.siblings
    if any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns)
]
total = sum(int(item.size or 0) for item in selected)
if not selected or total <= 0:
    raise SystemExit("mirror returned no downloadable checkpoint files")
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
    "insufficient disk space: available {:.2f} GiB, require {:.2f} GiB"
    .format(available / gib, required / gib)
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

printf 'Proxy environment: cleared\n'
printf 'Mirror endpoint:   %s\n' "${HF_ENDPOINT}"
printf 'Repository:        %s\n' "${MODEL_REPO}"
printf 'Revision:          %s\n' "${RESOLVED_REVISION}"
printf 'Destination:       %s\n' "${MODEL_DIR}"
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

"${DOWNLOAD_COMMAND[@]}"
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
}
path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

printf 'Download and validation completed: %s\n' "${MODEL_DIR}"
