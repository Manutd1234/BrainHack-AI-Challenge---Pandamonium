# Dockerfile for building the NLP image.


# The base image, an example deep learning VM.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download all models during build (no internet at runtime).
# 1. Sentence-transformer for embedding (22MB)
# 2. Cross-encoder for reranking (22MB)
# 3. Qwen2.5-3B-Instruct-AWQ for answer generation (~2GB, 4-bit quantized)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Embedding models downloaded')"

RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct-AWQ'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct-AWQ'); \
print('LLM downloaded')"

# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn nlp_server:app --port 5004 --host 0.0.0.0
