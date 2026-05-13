#!/usr/bin/env bash
set -euo pipefail

: "${ENABLE_LOCAL_VLLM:=1}"
: "${QWEN_MODEL:=Qwen/Qwen3.6-27B}"
: "${QWEN_PORT:=8000}"
: "${QWEN_TP_SIZE:=2}"
: "${QWEN_MAX_MODEL_LEN:=131072}"
: "${QWEN_EXTRA_ARGS:=}"

export QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:${QWEN_PORT}/v1}"

if [[ "${ENABLE_LOCAL_VLLM}" == "1" ]]; then
    echo "Starting vLLM for ${QWEN_MODEL} on port ${QWEN_PORT} with TP=${QWEN_TP_SIZE}"
    vllm serve "${QWEN_MODEL}" \
        --host 0.0.0.0 \
        --port "${QWEN_PORT}" \
        --tensor-parallel-size "${QWEN_TP_SIZE}" \
        --max-model-len "${QWEN_MAX_MODEL_LEN}" \
        --reasoning-parser qwen3 \
        --language-model-only \
        ${QWEN_EXTRA_ARGS} &
fi

exec uvicorn nlp_server:app --port 5004 --host 0.0.0.0
