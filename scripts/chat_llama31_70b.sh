#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASCADE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHAT_CHECKPOINT="${CASCADE_CHAT_CHECKPOINT:-/ssd/cascade-llm/models/Llama-3.1-70B-Instruct-W8A8}"
CHAT_GPU="${CASCADE_CHAT_GPU:-0}"
CHAT_SESSION="${CASCADE_CHAT_SESSION:-${CASCADE_ROOT}/real_results/70b_int8/chat_session.json}"
CHAT_TRANSCRIPT="${CASCADE_CHAT_TRANSCRIPT:-${CASCADE_ROOT}/real_results/70b_int8/chat_transcript.jsonl}"

cd "${CASCADE_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${CHAT_GPU}" \
    "${CASCADE_ROOT}/.venv/bin/python" \
    "${CASCADE_ROOT}/tools/chat_llama31_70b_int8.py" \
    --checkpoint "${CHAT_CHECKPOINT}" \
    --weight-store full_pinned \
    --granularity matrix \
    --slots 2 \
    --session "${CHAT_SESSION}" \
    --transcript "${CHAT_TRANSCRIPT}" \
    "$@"
