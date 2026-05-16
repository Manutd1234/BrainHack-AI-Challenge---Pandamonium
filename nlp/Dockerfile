# Dockerfile for the TIL NLP container.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    QWEN_MODEL_PATH=/app/src/qwen-quantized \
    QWEN_ENABLE_THINKING=0 \
    NLP_USE_DENSE=0 \
    NLP_USE_LLM=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade "transformers>=4.51.0"

COPY src/ ./src/
RUN test -f /app/src/qwen-quantized/config.json || \
    (echo "Missing /app/src/qwen-quantized/config.json. Run quantize.py or place the quantized Qwen3 model in nlp/src/qwen-quantized before docker build." && false)

WORKDIR /app/src

EXPOSE 5004
CMD ["uvicorn", "nlp_server:app", "--host", "0.0.0.0", "--port", "5004", "--workers", "1"]
