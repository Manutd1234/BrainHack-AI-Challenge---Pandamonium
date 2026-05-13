# Dockerfile for building the NLP image.


# The base image, an example deep learning VM.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV ENABLE_LOCAL_VLLM=1
ENV QWEN_MODEL=Qwen/Qwen3.6-27B
ENV QWEN_TP_SIZE=2
ENV QWEN_MAX_MODEL_LEN=131072
ENV QWEN_BASE_URL=http://127.0.0.1:8000/v1
ENV NLP_EMBEDDING_MODEL=BAAI/bge-m3
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BGE-M3 during build. For final offline NLP images, pass
# --build-arg DOWNLOAD_QWEN=1 to bundle the Qwen3.6 weights as well.
ARG DOWNLOAD_QWEN=0
COPY download_models.py .
RUN if [ "${DOWNLOAD_QWEN}" = "1" ]; then \
        python download_models.py; \
    else \
        python download_models.py --skip-qwen; \
    fi

# Copies your source files.
COPY src .
COPY start_server.sh .
RUN chmod +x /workspace/start_server.sh

# Starts your model server.
CMD ["/workspace/start_server.sh"]
