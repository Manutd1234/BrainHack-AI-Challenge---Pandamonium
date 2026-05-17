# Dockerfile for the TIL NLP container.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    QWEN_MODEL_PATH=/app/src/qwen-quantized \
    QWEN_ENABLE_THINKING=0 \
    NLP_USE_APPROX_LOOKUP=1 \
    NLP_APPROX_MIN_JACCARD=0.56 \
    NLP_APPROX_MIN_OVERLAP=5 \
    NLP_USE_DENSE=0 \
    NLP_USE_LLM=0 \
    NLP_LLM_MODE=selective \
    NLP_MAX_NEW_TOKENS=96

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade "transformers>=4.51.0"

COPY src/ ./src/

WORKDIR /app/src

EXPOSE 5004
CMD ["uvicorn", "nlp_server:app", "--host", "0.0.0.0", "--port", "5004", "--workers", "1"]
