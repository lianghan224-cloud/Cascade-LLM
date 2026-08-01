#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-runtime}"
CASCADE_HOME="${CASCADE_HOME:-/opt/cascade}"
CASCADE_VENV="${CASCADE_VENV:-/opt/cascade/venv}"

"${CASCADE_VENV}/bin/python" -m compileall -q \
  "${CASCADE_HOME}/layer_streaming"
"${CASCADE_VENV}/bin/cascade" --version
"${CASCADE_VENV}/bin/cascade" doctor --mode quick

if [[ "${MODE}" == "build" ]]; then
  "${CASCADE_VENV}/bin/python" -m unittest discover \
    -s "${CASCADE_HOME}/tests" \
    -p 'test_*docker*.py'
fi
