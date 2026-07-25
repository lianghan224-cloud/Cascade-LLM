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

export CASCADE_ROOT="${CASCADE_ROOT}"
export CASCADE_STORAGE_ROOT="${CASCADE_STORAGE_ROOT:-/ssd/cascade-llm}"
export CASCADE_MODEL_ROOT="${CASCADE_MODEL_ROOT:-${CASCADE_STORAGE_ROOT}/models}"
export CASCADE_LLAMA31_8B="${CASCADE_LLAMA31_8B:-${CASCADE_MODEL_ROOT}/Llama-3.1-8B}"
export HF_HOME="${HF_HOME:-${CASCADE_STORAGE_ROOT}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CASCADE_STORAGE_ROOT}/uv-cache}"
export TMPDIR="${TMPDIR:-${CASCADE_STORAGE_ROOT}/tmp}"

mkdir -p \
    "${CASCADE_MODEL_ROOT}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${UV_CACHE_DIR}" \
    "${TMPDIR}"

if [[ -d "${CASCADE_ROOT}/.venv" ]]; then
    export VIRTUAL_ENV="${CASCADE_ROOT}/.venv"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

printf 'CASCADE_ROOT=%s\n' "${CASCADE_ROOT}"
printf 'CASCADE_LLAMA31_8B=%s\n' "${CASCADE_LLAMA31_8B}"
printf 'HF_HOME=%s\n' "${HF_HOME}"
