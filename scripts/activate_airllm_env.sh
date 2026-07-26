#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASCADE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export AIRLLM_STORAGE_ROOT="${AIRLLM_STORAGE_ROOT:-/ssd/cascade-llm}"
export AIRLLM_CHECKOUT="${AIRLLM_CHECKOUT:-${AIRLLM_STORAGE_ROOT}/third_party/airllm}"
export AIRLLM_VENV="${AIRLLM_VENV:-${AIRLLM_STORAGE_ROOT}/venvs/airllm}"
export AIRLLM_MODEL_PATH="${AIRLLM_MODEL_PATH:-${AIRLLM_STORAGE_ROOT}/models/Llama-3.1-8B}"
export HF_HOME="${HF_HOME:-${AIRLLM_STORAGE_ROOT}/hf-cache-airllm}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset PIP_PROXY

if [[ ! -x "${AIRLLM_VENV}/bin/python" ]]; then
    printf 'ERROR: AirLLM environment is missing. Run %s first.\n' \
        "${CASCADE_ROOT}/scripts/install_airllm_no_proxy.sh" >&2
    return 1 2>/dev/null || exit 1
fi

export VIRTUAL_ENV="${AIRLLM_VENV}"
export PATH="${AIRLLM_VENV}/bin:${PATH}"

printf 'AIRLLM_CHECKOUT=%s\n' "${AIRLLM_CHECKOUT}"
printf 'AIRLLM_VENV=%s\n' "${AIRLLM_VENV}"
printf 'AIRLLM_MODEL_PATH=%s\n' "${AIRLLM_MODEL_PATH}"
