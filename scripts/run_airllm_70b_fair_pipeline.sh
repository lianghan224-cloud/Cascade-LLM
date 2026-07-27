#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${PROJECT_ROOT}/real_results/70b_int8/airllm_compression"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-30}"
export DOWNLOAD_WORKERS

mkdir -p "${RESULTS}"
echo "[$(date --iso-8601=seconds)] Download/resume BF16 checkpoint"
bash "${PROJECT_ROOT}/scripts/download_llama31_70b_bf16_cn.sh"

echo "[$(date --iso-8601=seconds)] Build AirLLM 8-bit shards and benchmark"
bash "${PROJECT_ROOT}/scripts/run_airllm_70b_cpu_resident_compression.sh"

echo "[$(date --iso-8601=seconds)] Summarize comparison"
"${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/benchmarks/summarize_airllm_70b_compression.py"
echo "[$(date --iso-8601=seconds)] Pipeline completed"
