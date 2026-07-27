#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRLLM_ROOT="${AIRLLM_ROOT:-/ssd/cascade-llm/third_party/airllm}"
AIRLLM_PYTHON="${AIRLLM_PYTHON:-/ssd/cascade-llm/venvs/airllm/bin/python}"
MODEL_DIR="${MODEL_DIR:-/disk2/home/guest/lianghan/models/Llama-3.1-70B-Instruct-BF16}"
SHARDS_ROOT="${SHARDS_ROOT:-/disk2/home/guest/lianghan/models/AirLLM-Llama-3.1-70B-Instruct-8bit}"
RESULTS="${RESULTS:-${PROJECT_ROOT}/real_results/70b_int8/airllm_compression}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_PINNED="${RUN_PINNED:-1}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${AIRLLM_ROOT}/air_llm${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -x "${AIRLLM_PYTHON}" ]]; then
    echo "Missing AirLLM Python: ${AIRLLM_PYTHON}" >&2
    exit 1
fi
for required in config.json model.safetensors.index.json tokenizer.json; do
    if [[ ! -s "${MODEL_DIR}/${required}" ]]; then
        echo "Incomplete BF16 checkpoint; missing ${MODEL_DIR}/${required}" >&2
        exit 1
    fi
done

mkdir -p "${SHARDS_ROOT}" "${RESULTS}"

"${AIRLLM_PYTHON}" -u \
    "${PROJECT_ROOT}/benchmarks/airllm_cpu_resident_compression_benchmark.py" \
    --checkpoint "${MODEL_DIR}" \
    --layer-shards-root "${SHARDS_ROOT}" \
    --compression 8bit \
    --prompt "The future of AI is" \
    --warmup-decode 0 \
    --decode-repeats 3 \
    --profile-repeats 2 \
    --output "${RESULTS}/bench_airllm_8bit_cpu_pageable.json" \
    2>&1 | tee "${RESULTS}/bench_airllm_8bit_cpu_pageable.log"

if [[ "${RUN_PINNED}" == "1" ]]; then
    if ! "${AIRLLM_PYTHON}" -u \
        "${PROJECT_ROOT}/benchmarks/airllm_cpu_resident_compression_benchmark.py" \
        --checkpoint "${MODEL_DIR}" \
        --layer-shards-root "${SHARDS_ROOT}" \
        --compression 8bit \
        --pin-cpu-cache \
        --prompt "The future of AI is" \
        --warmup-decode 0 \
        --decode-repeats 3 \
        --profile-repeats 2 \
        --output "${RESULTS}/bench_airllm_8bit_cpu_pinned.json" \
        2>&1 | tee "${RESULTS}/bench_airllm_8bit_cpu_pinned.log"; then
        echo "Pinned CPU cache run failed; pageable result remains valid." \
            | tee "${RESULTS}/bench_airllm_8bit_cpu_pinned.failed"
    fi
fi
