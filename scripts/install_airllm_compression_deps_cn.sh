#!/usr/bin/env bash
set -Eeuo pipefail

AIRLLM_PYTHON="${AIRLLM_PYTHON:-/ssd/cascade-llm/venvs/airllm/bin/python}"
PYPI_INDEX="${PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}"

unset http_proxy https_proxy all_proxy ftp_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY

if [[ ! -x "${AIRLLM_PYTHON}" ]]; then
    echo "Missing AirLLM Python: ${AIRLLM_PYTHON}" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required" >&2
    exit 1
fi

uv pip install \
    --python "${AIRLLM_PYTHON}" \
    --index-url "${PYPI_INDEX}" \
    "bitsandbytes==0.45.5"

"${AIRLLM_PYTHON}" - <<'PY'
import bitsandbytes
import torch

print("bitsandbytes", bitsandbytes.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
