#!/usr/bin/env bash
set -euo pipefail

# Install AirLLM into an isolated, persistent environment without using any
# proxy inherited from the current SSH session. This does not modify the
# Cascade-LLM .venv and does not download or split model weights.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASCADE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

AIRLLM_COMMIT="${AIRLLM_COMMIT:-17677cb821016b36a0610c8e1f2befab030d1942}"
AIRLLM_STORAGE_ROOT="${AIRLLM_STORAGE_ROOT:-/ssd/cascade-llm}"
AIRLLM_CHECKOUT="${AIRLLM_CHECKOUT:-${AIRLLM_STORAGE_ROOT}/third_party/airllm}"
AIRLLM_VENV="${AIRLLM_VENV:-${AIRLLM_STORAGE_ROOT}/venvs/airllm}"
AIRLLM_PYTHON_ROOT="${AIRLLM_PYTHON_ROOT:-${AIRLLM_STORAGE_ROOT}/python}"
AIRLLM_UV_CACHE="${AIRLLM_UV_CACHE:-${AIRLLM_STORAGE_ROOT}/uv-cache-airllm}"
AIRLLM_MODEL_PATH="${AIRLLM_MODEL_PATH:-${AIRLLM_STORAGE_ROOT}/models/Llama-3.1-8B}"
AIRLLM_INSTALL_RECEIPT="${AIRLLM_INSTALL_RECEIPT:-${CASCADE_ROOT}/real_results/airllm/install_receipt.json}"
AIRLLM_PYPI_INDEX="${AIRLLM_PYPI_INDEX:-https://pypi.org/simple}"
AIRLLM_TORCH_INDEX="${AIRLLM_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
AIRLLM_TORCH_WHEEL_URL="${AIRLLM_TORCH_WHEEL_URL:-}"

# Clear both conventional and lowercase proxy variables. Git commands below
# also override configured HTTP proxies for this invocation.
unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset PIP_PROXY

# Ignore user pip configuration and select indexes explicitly below.
export PIP_CONFIG_FILE=/dev/null
export UV_CACHE_DIR="${AIRLLM_UV_CACHE}"
export UV_PYTHON_INSTALL_DIR="${AIRLLM_PYTHON_ROOT}"
export UV_LINK_MODE=copy

if ! command -v uv >/dev/null 2>&1; then
    printf 'ERROR: uv is required but was not found in PATH.\n' >&2
    exit 1
fi

if [[ ! -f "${AIRLLM_MODEL_PATH}/config.json" ]]; then
    printf 'ERROR: model config is missing: %s/config.json\n' "${AIRLLM_MODEL_PATH}" >&2
    exit 1
fi

shopt -s nullglob
MODEL_SHARDS=("${AIRLLM_MODEL_PATH}"/model-*.safetensors)
shopt -u nullglob
if [[ "${#MODEL_SHARDS[@]}" -ne 4 ]]; then
    printf 'ERROR: expected 4 model safetensor shards, found %s in %s\n' \
        "${#MODEL_SHARDS[@]}" "${AIRLLM_MODEL_PATH}" >&2
    exit 1
fi

mkdir -p \
    "$(dirname -- "${AIRLLM_CHECKOUT}")" \
    "$(dirname -- "${AIRLLM_VENV}")" \
    "${AIRLLM_PYTHON_ROOT}" \
    "${AIRLLM_UV_CACHE}" \
    "$(dirname -- "${AIRLLM_INSTALL_RECEIPT}")"

printf 'Proxy-free AirLLM installation\n'
printf '  source commit: %s\n' "${AIRLLM_COMMIT}"
printf '  source checkout: %s\n' "${AIRLLM_CHECKOUT}"
printf '  virtual environment: %s\n' "${AIRLLM_VENV}"
printf '  model checkpoint: %s\n' "${AIRLLM_MODEL_PATH}"
printf '  Python package index: %s\n' "${AIRLLM_PYPI_INDEX}"
printf '  PyTorch package index: %s\n' "${AIRLLM_TORCH_INDEX}"
if [[ -n "${AIRLLM_TORCH_WHEEL_URL}" ]]; then
    printf '  PyTorch wheel override: %s\n' "${AIRLLM_TORCH_WHEEL_URL}"
fi

GIT_DIRECT=(git -c http.proxy= -c https.proxy=)
"${GIT_DIRECT[@]}" ls-remote https://github.com/lyogavin/airllm.git "${AIRLLM_COMMIT}" >/dev/null

if [[ ! -e "${AIRLLM_CHECKOUT}" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 "${GIT_DIRECT[@]}" clone \
        --filter=blob:none \
        https://github.com/lyogavin/airllm.git \
        "${AIRLLM_CHECKOUT}"
elif [[ ! -d "${AIRLLM_CHECKOUT}/.git" ]]; then
    printf 'ERROR: checkout path exists but is not a Git repository: %s\n' \
        "${AIRLLM_CHECKOUT}" >&2
    exit 1
fi

# Version 1 of this installer used `git clone --no-checkout`. Such a new
# checkout has a valid HEAD and index but an empty worktree, which Git reports
# as every tracked file being deleted. Recover exactly that installer-created
# state without accepting or overwriting a checkout that contains user files.
CHECKOUT_STATUS="$("${GIT_DIRECT[@]}" -C "${AIRLLM_CHECKOUT}" status --porcelain)"
CHECKOUT_TOP_ENTRY="$(
    find "${AIRLLM_CHECKOUT}" \
        -mindepth 1 \
        -maxdepth 1 \
        ! -name .git \
        -print \
        -quit
)"
if [[ -n "${CHECKOUT_STATUS}" && -z "${CHECKOUT_TOP_ENTRY}" ]] \
    && printf '%s\n' "${CHECKOUT_STATUS}" | awk '
        substr($0, 1, 2) != "D " && substr($0, 1, 2) != " D" {
            bad = 1
        }
        END {
            exit bad
        }
    '
then
    printf 'Recovering the empty worktree created by installer version 1.\n'
    "${GIT_DIRECT[@]}" -C "${AIRLLM_CHECKOUT}" restore \
        --source=HEAD \
        --staged \
        --worktree \
        :/
    CHECKOUT_STATUS="$("${GIT_DIRECT[@]}" -C "${AIRLLM_CHECKOUT}" status --porcelain)"
fi

if [[ -n "${CHECKOUT_STATUS}" ]]; then
    printf 'ERROR: AirLLM checkout has local modifications; refusing to overwrite them.\n' >&2
    exit 1
fi

"${GIT_DIRECT[@]}" -C "${AIRLLM_CHECKOUT}" fetch \
    --depth=1 \
    origin \
    "${AIRLLM_COMMIT}"
"${GIT_DIRECT[@]}" -C "${AIRLLM_CHECKOUT}" checkout \
    --detach \
    "${AIRLLM_COMMIT}"

uv python install 3.11
if [[ ! -x "${AIRLLM_VENV}/bin/python" ]]; then
    uv venv --python 3.11 "${AIRLLM_VENV}"
fi

# Install the same CUDA-enabled PyTorch major/minor used by the Cascade
# experiment. Some domestic mirrors host the CUDA wheel but omit its local
# version (`+cu121`) from their PEP 503 index. In that case the wrapper passes
# the already-probed wheel URL directly, while dependencies still resolve
# through the selected PyPI mirror and reuse the persistent uv cache.
if [[ -n "${AIRLLM_TORCH_WHEEL_URL}" ]]; then
    uv pip install \
        --python "${AIRLLM_VENV}/bin/python" \
        --index-url "${AIRLLM_PYPI_INDEX}" \
        "${AIRLLM_TORCH_WHEEL_URL}"
else
    uv pip install \
        --python "${AIRLLM_VENV}/bin/python" \
        --index-url "${AIRLLM_PYPI_INDEX}" \
        --extra-index-url "${AIRLLM_TORCH_INDEX}" \
        --index-strategy unsafe-best-match \
        "torch==2.4.1+cu121"
fi

# Use a stable Transformers 4.x API surface. AirLLM 3.0.1 declares
# transformers>=4.49,<5.13, but a 4.x cap avoids introducing a future major API
# change into the comparison environment.
uv pip install \
    --python "${AIRLLM_VENV}/bin/python" \
    --index-url "${AIRLLM_PYPI_INDEX}" \
    "numpy<2" \
    "transformers>=4.49,<5" \
    "accelerate>=1,<2" \
    safetensors \
    huggingface-hub \
    scipy \
    sentencepiece \
    tqdm \
    psutil

uv pip install \
    --python "${AIRLLM_VENV}/bin/python" \
    --index-url "${AIRLLM_PYPI_INDEX}" \
    --no-deps \
    --editable "${AIRLLM_CHECKOUT}/air_llm"

export AIRLLM_COMMIT
export AIRLLM_CHECKOUT
export AIRLLM_VENV
export AIRLLM_MODEL_PATH
export AIRLLM_INSTALL_RECEIPT

"${AIRLLM_VENV}/bin/python" - <<'PY'
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import accelerate
import safetensors
import torch
import transformers
from transformers import AutoConfig

import airllm


model_path = Path(os.environ["AIRLLM_MODEL_PATH"])
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
shards = sorted(model_path.glob("model-*.safetensors"))
packages = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if name:
        packages[name] = distribution.version

receipt = {
    "schema_version": 1,
    "airllm_commit": os.environ["AIRLLM_COMMIT"],
    "airllm_checkout": os.environ["AIRLLM_CHECKOUT"],
    "airllm_venv": os.environ["AIRLLM_VENV"],
    "airllm_package_version": importlib.metadata.version("airllm"),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "accelerate_version": accelerate.__version__,
    "safetensors_version": safetensors.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ],
    "model_path": str(model_path),
    "model_type": config.model_type,
    "model_hidden_size": config.hidden_size,
    "model_layers": config.num_hidden_layers,
    "model_shards": [
        {"name": shard.name, "size_bytes": shard.stat().st_size}
        for shard in shards
    ],
    "proxy_variables_present": sorted(
        key
        for key in (
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        )
        if os.environ.get(key)
    ),
    "installed_packages": dict(sorted(packages.items())),
}

if not receipt["cuda_available"]:
    raise SystemExit("CUDA is unavailable in the AirLLM environment")
if len(shards) != 4:
    raise SystemExit("checkpoint shard validation failed")

receipt_path = Path(os.environ["AIRLLM_INSTALL_RECEIPT"])
receipt_path.write_text(
    json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(receipt, indent=2, ensure_ascii=False))
PY

printf '\nAirLLM installation completed successfully.\n'
printf 'Receipt: %s\n' "${AIRLLM_INSTALL_RECEIPT}"
printf 'Activate with: source %s/scripts/activate_airllm_env.sh\n' "${CASCADE_ROOT}"
