# Dockerfile for the TIL NLP container.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    QWEN_MODEL_PATH=/app/src/qwen-quantized \
    QWEN_ENABLE_THINKING=0 \
    NLP_USE_APPROX_LOOKUP=1 \
    NLP_USE_QA_MEMORY=0 \
    NLP_USE_QA_MEMORY_HINTS=0 \
    NLP_APPROX_MIN_JACCARD=0.48 \
    NLP_APPROX_MIN_OVERLAP=4 \
    NLP_APPROX_MIN_CONFIDENCE=0.64 \
    NLP_APPROX_HINT_MIN_CONFIDENCE=0.42 \
    NLP_QA_MEMORY_MIN_CONFIDENCE=0.62 \
    NLP_QA_MEMORY_HINT_CONFIDENCE=0.36 \
    NLP_QA_MEMORY_MIN_OVERLAP=4 \
    NLP_DOC_BM25_BOOST=0.55 \
    NLP_SENTENCE_WINDOW=3 \
    NLP_SENTENCE_STRIDE=2 \
    NLP_MAX_DOC_CHUNKS=18 \
    NLP_MAX_MODEL_LEN=2048 \
    NLP_LLM_CONTEXT_CHARS=3600 \
    NLP_LORA_ADAPTER_PATH=/app/src/lora_adapter \
    NLP_MAX_GPU_MEMORY=12GiB \
    NLP_USE_DENSE=1 \
    NLP_USE_LLM=0 \
    NLP_USE_QA_READER=0 \
    NLP_QA_READER_MODEL_PATH=/app/models/qa-reader \
    NLP_QA_READER_CONTEXTS=5 \
    NLP_QA_READER_MIN_SCORE=5.0 \
    NLP_LLM_MODE=selective \
    NLP_MAX_NEW_TOKENS=96 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade "transformers>=4.51.0"

# Pre-download BGE models to cache them for offline runtime evaluation.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3'); snapshot_download(repo_id='BAAI/bge-reranker-large')" \
    && chmod -R 777 /app/.cache

COPY src/ ./src/

WORKDIR /app/src

EXPOSE 5004
CMD ["uvicorn", "nlp_server:app", "--host", "0.0.0.0", "--port", "5004", "--workers", "1"]
