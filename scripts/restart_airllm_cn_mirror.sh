#!/usr/bin/env bash
set -euo pipefail

# Stop only the active AirLLM package transaction, retain or clean its
# regenerable package cache, and restart installation through Aliyun mirrors.
# Model weights and the pinned AirLLM source checkout are never removed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASCADE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:---reuse-cache}"
case "${MODE}" in
    --reuse-cache)
        ;;
    --clean)
        ;;
    *)
        printf 'Usage: %s [--reuse-cache|--clean]\n' "$0" >&2
        exit 2
        ;;
esac

AIRLLM_STORAGE_ROOT="${AIRLLM_STORAGE_ROOT:-/ssd/cascade-llm}"
AIRLLM_VENV="${AIRLLM_VENV:-${AIRLLM_STORAGE_ROOT}/venvs/airllm}"
AIRLLM_UV_CACHE="${AIRLLM_UV_CACHE:-${AIRLLM_STORAGE_ROOT}/uv-cache-airllm}"
AIRLLM_MODEL_PATH="${AIRLLM_MODEL_PATH:-${AIRLLM_STORAGE_ROOT}/models/Llama-3.1-8B}"
AIRLLM_INSTALL_LOG="${AIRLLM_INSTALL_LOG:-${CASCADE_ROOT}/real_results/airllm/install_cn.log}"

AIRLLM_PYPI_INDEX="${AIRLLM_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
AIRLLM_TORCH_INDEX="${AIRLLM_TORCH_INDEX:-https://mirrors.aliyun.com/pytorch-wheels/cu121}"
AIRLLM_TORCH_WHEEL_URL="${AIRLLM_TORCH_WHEEL_URL:-${AIRLLM_TORCH_INDEX}/torch-2.4.1%2Bcu121-cp311-cp311-linux_x86_64.whl}"

unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset PIP_PROXY
export PIP_CONFIG_FILE=/dev/null

if [[ ! -f "${AIRLLM_MODEL_PATH}/config.json" ]]; then
    printf 'ERROR: model checkpoint is missing: %s\n' "${AIRLLM_MODEL_PATH}" >&2
    exit 1
fi
if [[ "${AIRLLM_UV_CACHE}" != /ssd/cascade-llm/uv-cache-airllm ]]; then
    printf 'ERROR: refusing unexpected cache target: %s\n' "${AIRLLM_UV_CACHE}" >&2
    exit 1
fi
if [[ "${AIRLLM_VENV}" != /ssd/cascade-llm/venvs/airllm ]]; then
    printf 'ERROR: refusing unexpected environment target: %s\n' "${AIRLLM_VENV}" >&2
    exit 1
fi

mkdir -p "${AIRLLM_UV_CACHE}" "$(dirname -- "${AIRLLM_INSTALL_LOG}")"

mapfile -t ACTIVE_UV_PIDS < <(
    pgrep -f "^uv pip install --python ${AIRLLM_VENV}/bin/python" || true
)
if [[ "${#ACTIVE_UV_PIDS[@]}" -gt 0 ]]; then
    printf 'Interrupting active AirLLM uv transaction: %s\n' \
        "${ACTIVE_UV_PIDS[*]}"
    kill -INT "${ACTIVE_UV_PIDS[@]}"
    for _ in $(seq 1 15); do
        RUNNING=0
        for pid in "${ACTIVE_UV_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                RUNNING=1
            fi
        done
        if [[ "${RUNNING}" -eq 0 ]]; then
            break
        fi
        sleep 1
    done
    for pid in "${ACTIVE_UV_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            printf 'ERROR: process %s did not stop; interrupt it manually.\n' \
                "${pid}" >&2
            exit 1
        fi
    done
fi

# Wait briefly for the parent installer pipeline to release the environment
# lock after its uv child exits.
for _ in $(seq 1 10); do
    if ! pgrep -f "^bash -o pipefail ${CASCADE_ROOT}/scripts/install_airllm_no_proxy.sh" \
        >/dev/null 2>&1 \
        && ! pgrep -f "^bash -o pipefail scripts/install_airllm_no_proxy.sh" \
        >/dev/null 2>&1
    then
        break
    fi
    sleep 1
done

if [[ "${MODE}" == "--clean" ]]; then
    printf 'Clean mode: removing only the regenerable AirLLM venv and package cache.\n'
    if [[ -d "${AIRLLM_VENV}" ]]; then
        find "${AIRLLM_VENV}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
    fi
    if [[ -d "${AIRLLM_UV_CACHE}" ]]; then
        find "${AIRLLM_UV_CACHE}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
    fi
else
    printf 'Reuse mode: retaining completed cache entries and removing partial temp directories.\n'
    find "${AIRLLM_UV_CACHE}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name '.tmp*' \
        -exec rm -rf -- {} +
fi

printf 'Testing Aliyun PyTorch mirror with a 1 MiB range request...\n'
MIRROR_SPEED="$(
    curl \
        --noproxy '*' \
        --location \
        --range 0-1048575 \
        --connect-timeout 10 \
        --max-time 30 \
        --output /dev/null \
        --silent \
        --show-error \
        --write-out '%{speed_download}' \
        "${AIRLLM_TORCH_WHEEL_URL}"
)"
printf 'Aliyun probe speed: %s bytes/s\n' "${MIRROR_SPEED}"
if ! awk -v speed="${MIRROR_SPEED}" 'BEGIN {exit !(speed >= 1048576)}'; then
    printf 'ERROR: Aliyun mirror is below 1 MiB/s; refusing a slow restart.\n' >&2
    exit 1
fi

export AIRLLM_PYPI_INDEX
export AIRLLM_TORCH_INDEX
export AIRLLM_TORCH_WHEEL_URL
export AIRLLM_UV_CACHE

printf 'Restarting AirLLM installation with domestic mirrors.\n'
printf '  PyPI: %s\n' "${AIRLLM_PYPI_INDEX}"
printf '  PyTorch: %s\n' "${AIRLLM_TORCH_INDEX}"
printf '  PyTorch wheel: %s\n' "${AIRLLM_TORCH_WHEEL_URL}"
printf '  log: %s\n' "${AIRLLM_INSTALL_LOG}"

set -o pipefail
bash "${CASCADE_ROOT}/scripts/install_airllm_no_proxy.sh" \
    2>&1 | tee "${AIRLLM_INSTALL_LOG}"
