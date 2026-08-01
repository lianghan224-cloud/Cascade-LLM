#!/usr/bin/env bash
set -euo pipefail

CASCADE_HOME="${CASCADE_HOME:-/opt/cascade}"
CASCADE_VENV="${CASCADE_VENV:-/opt/cascade/venv}"

"${CASCADE_VENV}/bin/python" -m pip install \
  --no-deps \
  --no-build-isolation \
  "${CASCADE_HOME}"

"${CASCADE_VENV}/bin/python" -c \
  'import layer_streaming; print(layer_streaming.__version__)'

rm -rf -- \
  "${CASCADE_HOME}/build" \
  "${CASCADE_HOME}/cascade_llm.egg-info"
