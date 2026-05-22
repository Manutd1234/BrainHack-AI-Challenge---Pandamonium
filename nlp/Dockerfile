FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        python3.10 python3-pip python3.10-dev git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install PyTorch with CUDA 11.8 support explicitly to prevent compatibility issues
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models at build time to ensure zero network requests at runtime (Offline Mode)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
print('BGE-M3...'); snapshot_download('BAAI/bge-m3', local_dir='/app/model/bge-m3'); \
print('Reranker...'); snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='/app/model/bge-reranker'); \
print('DeBERTa...'); snapshot_download('deepset/deberta-v3-large-squad2', local_dir='/app/model/deberta-reader'); \
print('Done')"

ENV EMBED_MODEL_PATH=/app/model/bge-m3
ENV RERANKER_MODEL_PATH=/app/model/bge-reranker
ENV READER_MODEL_PATH=/app/model/deberta-reader
# QA training pairs — copy nlp.jsonl to data/nlp.jsonl before building
ENV QA_JSONL_PATH=/app/data/nlp.jsonl
# Tunable thresholds (no rebuild needed)
ENV QA_CACHE_THRESHOLD=0.92
ENV RERANK_L4_THRESHOLD=0.28
ENV EXTRACT_HIGH=0.65
ENV EXTRACT_LOW=0.18
ENV W_DENSE=0.40
ENV W_SPARSE=0.30
ENV W_COLBERT=0.30
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TOKENIZERS_PARALLELISM=false

COPY data/ /app/data/
COPY src/  ./src/
WORKDIR /app/src
EXPOSE 5004
CMD ["uvicorn", "nlp_server:app", "--host", "0.0.0.0", "--port", "5004", \
     "--workers", "1", "--timeout-keep-alive", "300"]
