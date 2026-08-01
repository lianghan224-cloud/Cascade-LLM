#!/usr/bin/env bash
set -euo pipefail

BUNDLE_NAME="${1:-generic}"
CASCADE_HOME="${CASCADE_HOME:-/opt/cascade}"
CASCADE_VENV="${CASCADE_VENV:-/opt/cascade/venv}"
VERSIONS_FILE="${CASCADE_HOME}/docker/versions.env"
BUNDLE_FILE="${CASCADE_HOME}/docker/provider-bundles/${BUNDLE_NAME}.json"
WHEEL_ROOT="${CASCADE_HOME}/provider-wheels"

if [[ ! -f "${BUNDLE_FILE}" ]]; then
  printf 'unknown provider bundle: %s\n' "${BUNDLE_NAME}" >&2
  exit 2
fi

set -a
# shellcheck source=/dev/null
source "${VERSIONS_FILE}"
set +a
mkdir -p "${WHEEL_ROOT}"

mapfile -t PROVIDERS < <(
  "${CASCADE_VENV}/bin/python" - "${BUNDLE_FILE}" <<'PY'
import json
import sys

from layer_streaming.container_contracts import ProviderBundleManifest

manifest = ProviderBundleManifest.read(sys.argv[1])
for provider in manifest.providers:
    print(json.dumps(provider.__dict__, sort_keys=True))
PY
)

INSTALLED='[]'
for provider_json in "${PROVIDERS[@]}"; do
  readarray -t FIELDS < <(
    "${CASCADE_VENV}/bin/python" - "${provider_json}" <<'PY'
import json
import sys
value = json.loads(sys.argv[1])
for key in ('package', 'source', 'binary', 'binary_target', 'build_metadata',
            'numerical_contract', 'contract_target', 'provider_abi',
            'architecture'):
    print(value[key])
PY
  )
  PACKAGE="${FIELDS[0]}"
  SOURCE="${CASCADE_HOME}/${FIELDS[1]}"
  BINARY="${CASCADE_HOME}/${FIELDS[2]}"
  BINARY_TARGET="${FIELDS[3]}"
  BUILD_METADATA="${CASCADE_HOME}/${FIELDS[4]}"
  CONTRACT="${CASCADE_HOME}/${FIELDS[5]}"
  CONTRACT_TARGET="${FIELDS[6]}"
  ABI="${FIELDS[7]}"
  ARCHITECTURE="${FIELDS[8]}"
  if [[ ! -f "${BINARY}" ]]; then
    printf 'qualified provider binary is missing: %s\n' "${BINARY}" >&2
    exit 3
  fi
  if [[ ! -f "${BUILD_METADATA}" || ! -f "${CONTRACT}" ]]; then
    printf 'provider metadata/contract is missing for %s\n' "${PACKAGE}" >&2
    exit 3
  fi
  if readelf -d "${BINARY}" | grep -E '(RPATH|RUNPATH)' | grep -q '\[/'; then
    printf 'provider binary contains a host-absolute RPATH/RUNPATH: %s\n' \
      "${BINARY}" >&2
    printf 'rebuild it with the container-safe provider build script\n' >&2
    exit 3
  fi
  "${CASCADE_VENV}/bin/python" - \
    "${BUILD_METADATA}" "${ABI}" "${ARCHITECTURE}" <<'PY'
import sys
from layer_streaming.hardware import ProviderBuildMetadata
metadata = ProviderBuildMetadata.read(sys.argv[1])
expected_abi = int(sys.argv[2])
architecture = sys.argv[3]
if metadata.abi != expected_abi:
    raise SystemExit('provider ABI mismatch')
if architecture not in metadata.compiled_architectures:
    raise SystemExit('provider compiled architecture mismatch')
PY
  PACKAGE_ROOT="${SOURCE}/cascade_provider"
  mkdir -p \
    "${PACKAGE_ROOT}/$(dirname -- "${BINARY_TARGET}")" \
    "${PACKAGE_ROOT}/$(dirname -- "${CONTRACT_TARGET}")"
  install -m 0755 "${BINARY}" \
    "${PACKAGE_ROOT}/${BINARY_TARGET}"
  install -m 0644 "${BUILD_METADATA}" \
    "${PACKAGE_ROOT}/build_metadata.json"
  install -m 0644 "${CONTRACT}" \
    "${PACKAGE_ROOT}/${CONTRACT_TARGET}"
  "${CASCADE_VENV}/bin/python" -m pip wheel \
    --no-deps --no-build-isolation --wheel-dir "${WHEEL_ROOT}" "${SOURCE}"
  WHEEL=$(find "${WHEEL_ROOT}" -maxdepth 1 -type f \
    -name "${PACKAGE}-*.whl" -print -quit)
  if [[ -z "${WHEEL}" ]]; then
    printf 'provider wheel was not produced for %s\n' "${PACKAGE}" >&2
    exit 3
  fi
  "${CASCADE_VENV}/bin/python" -m pip install --no-deps "${WHEEL}"
  rm -rf -- "${SOURCE}/build"
  find "${SOURCE}" -maxdepth 1 -type d -name '*.egg-info' \
    -exec rm -rf -- {} +
  INSTALLED=$("${CASCADE_VENV}/bin/python" - \
    "${INSTALLED}" "${provider_json}" <<'PY'
import json
import sys
items = json.loads(sys.argv[1])
items.append(json.loads(sys.argv[2]))
print(json.dumps(items, sort_keys=True))
PY
  )
done

printf '%s\n' "${INSTALLED}" > "${CASCADE_HOME}/installed-providers.json"
