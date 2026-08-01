#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  "${CASCADE_CACHE_ROOT:-/cache/cascade}" \
  "${CASCADE_RESULT_ROOT:-/results}" \
  "${HF_HOME:-/cache/huggingface}"

COMMAND="${1:-doctor}"
case "${COMMAND}" in
  doctor|inspect|qualify|shell)
    ;;
  *)
    if [[ "${CASCADE_RUN_PREFLIGHT:-1}" == "1" ]]; then
      PREFLIGHT_FILE="${TMPDIR:-/tmp}/cascade-entrypoint-preflight.json"
      if ! cascade doctor --mode quick --require-cuda \
        --output "${PREFLIGHT_FILE}" >/dev/null; then
        cat "${PREFLIGHT_FILE}" >&2 || true
        exit 1
      fi
    fi
    ;;
esac

exec cascade "$@"
