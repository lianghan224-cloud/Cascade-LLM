#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${TMPDIR:-/tmp}/cascade-healthcheck.json"
cascade doctor --mode quick --require-cuda --output "${OUTPUT}" >/dev/null
