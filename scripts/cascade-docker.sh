#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASCADE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VERSIONS_FILE="${CASCADE_ROOT}/docker/versions.env"

set -a
# shellcheck source=/dev/null
source "${VERSIONS_FILE}"
set +a

export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"
export CASCADE_GIT_COMMIT="${CASCADE_GIT_COMMIT:-$(git -C "${CASCADE_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || printf unverified)}"
CUDA_TAG="${CUDA_VERSION//./}"
TORCH_TAG="${PYTORCH_VERSION//./}"
export CASCADE_IMAGE_TYPE="${CASCADE_IMAGE_TYPE:-runtime}"
export CASCADE_IMAGE="${CASCADE_IMAGE:-cascade-llm:${CASCADE_VERSION}-${CASCADE_IMAGE_TYPE}-cu${CUDA_TAG}-torch${TORCH_TAG}}"

COMPOSE=(docker compose --env-file "${VERSIONS_FILE}" -f "${CASCADE_ROOT}/compose.yaml")
ACTION="${1:-doctor}"
if (($#)); then
  shift
fi

ensure_directories() {
  mkdir -p \
    "${CASCADE_MODEL_DIR:-${CASCADE_ROOT}/models}" \
    "${CASCADE_CONFIG_DIR:-${CASCADE_ROOT}/config}" \
    "${CASCADE_CACHE_DIR:-${CASCADE_ROOT}/cache}" \
    "${CASCADE_RESULT_DIR:-${CASCADE_ROOT}/results}"
}

case "${ACTION}" in
  install)
    command -v docker >/dev/null || {
      printf 'Docker is not installed.\n' >&2
      exit 2
    }
    docker compose version >/dev/null
    ensure_directories
    printf 'Cascade Docker host directories are ready.\n'
    ;;
  pull)
    "${COMPOSE[@]}" pull "$@"
    ;;
  build)
    ensure_directories
    "${COMPOSE[@]}" build "$@"
    ;;
  doctor|validate|run|chat|benchmark|qualify|quantize)
    ensure_directories
    "${COMPOSE[@]}" run --rm cascade "${ACTION}" "$@"
    ;;
  shell)
    ensure_directories
    "${COMPOSE[@]}" run --rm cascade shell
    ;;
  dev-shell)
    ensure_directories
    "${COMPOSE[@]}" -f "${CASCADE_ROOT}/compose.dev.yaml" \
      run --rm cascade shell
    ;;
  clean)
    "${COMPOSE[@]}" down --remove-orphans
    ;;
  *)
    printf 'Unknown action: %s\n' "${ACTION}" >&2
    exit 2
    ;;
esac
