#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASCADE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${CASCADE_PYTHON_BIN:-python3.10}"
VENV_DIR="${CASCADE_VENV_DIR:-${CASCADE_ROOT}/.venv}"
VERSIONS_FILE="${CASCADE_ROOT}/docker/versions.env"
LOCK_FILE="${CASCADE_ROOT}/requirements.lock"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  printf 'Missing %s. Install Python 3.10 and its venv package first.\n' "${PYTHON_BIN}" >&2
  exit 2
}

PYTHON_VERSION_ACTUAL="$(${PYTHON_BIN} -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')"
if [[ "${PYTHON_VERSION_ACTUAL}" != "3.10" ]]; then
  printf 'Refusing Python %s; the reproducible server baseline is Python 3.10.\n' "${PYTHON_VERSION_ACTUAL}" >&2
  printf 'Set CASCADE_PYTHON_BIN to a Python 3.10 executable.\n' >&2
  exit 2
fi

if [[ -e "${VENV_DIR}" ]]; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    printf 'Existing path is not a usable virtual environment: %s\n' "${VENV_DIR}" >&2
    printf 'Move it aside manually; this script will not delete it.\n' >&2
    exit 2
  fi
  EXISTING_VERSION="$(${VENV_DIR}/bin/python -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')"
  if [[ "${EXISTING_VERSION}" != "3.10" ]]; then
    printf 'Existing venv uses Python %s, expected 3.10: %s\n' "${EXISTING_VERSION}" "${VENV_DIR}" >&2
    printf 'Move it aside manually and rerun.\n' >&2
    exit 2
  fi
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

set -a
# shellcheck source=/dev/null
source "${VERSIONS_FILE}"
set +a

"${VENV_DIR}/bin/python" -m ensurepip --upgrade
"${VENV_DIR}/bin/python" -m pip install --upgrade \
  "pip==${PIP_VERSION}" \
  "setuptools==${SETUPTOOLS_VERSION}" \
  "wheel==${WHEEL_VERSION}"
"${VENV_DIR}/bin/python" -m pip install -r "${LOCK_FILE}"
"${VENV_DIR}/bin/python" -m pip install --no-deps --no-build-isolation --editable "${CASCADE_ROOT}"
"${VENV_DIR}/bin/python" -m pip check

"${VENV_DIR}/bin/python" - <<'PY'
import torch
import transformers
import safetensors
import huggingface_hub
import layer_streaming

print("cascade import: ok")
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("safetensors:", safetensors.__version__)
print("huggingface-hub:", huggingface_hub.__version__)
print("cascade:", getattr(layer_streaming, "__version__", "unknown"))
PY

mkdir -p "${CASCADE_ROOT}/reports/migration"
"${VENV_DIR}/bin/python" "${CASCADE_ROOT}/tools/capture_environment.py" \
  --output "${CASCADE_ROOT}/reports/migration/environment.json"

printf '\nBootstrap complete. Activate with:\n  source %s/bin/activate\n' "${VENV_DIR}"
printf 'Next run:\n  bash %s/scripts/verify_server.sh\n' "${CASCADE_ROOT}"
