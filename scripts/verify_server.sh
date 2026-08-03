#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASCADE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${CASCADE_VENV_DIR:-${CASCADE_ROOT}/.venv}"
PYTHON="${VENV_DIR}/bin/python"
OUTPUT_DIR="${CASCADE_MIGRATION_REPORT_DIR:-${CASCADE_ROOT}/reports/migration}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Missing virtual environment: %s\n' "${VENV_DIR}" >&2
  printf 'Run CASCADE_PYTHON_BIN=python3.10 bash scripts/bootstrap_server.sh first.\n' >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${CASCADE_ROOT}/tools/capture_environment.py" \
  --output "${OUTPUT_DIR}/environment.json"
"${PYTHON}" -m pip check
"${PYTHON}" -m compileall -q "${CASCADE_ROOT}/layer_streaming"

"${PYTHON}" -m layer_streaming.cli doctor \
  --mode quick --require-cuda \
  --output "${OUTPUT_DIR}/doctor.json"
"${PYTHON}" -m layer_streaming.cli inspect \
  --output "${OUTPUT_DIR}/hardware.json"

set +e
"${PYTHON}" -m unittest discover \
  -s "${CASCADE_ROOT}/tests" -p 'test_*.py' -q \
  >"${OUTPUT_DIR}/unittest.log" 2>&1
TEST_STATUS=$?
set -e
cat "${OUTPUT_DIR}/unittest.log"
if ((TEST_STATUS != 0)); then
  printf 'Server verification failed: unit tests exited %d.\n' "${TEST_STATUS}" >&2
  exit "${TEST_STATUS}"
fi

printf '\nServer verification passed. Local reports:\n  %s\n' "${OUTPUT_DIR}"
printf 'Review GPU architecture before selecting an architecture-specific provider.\n'
