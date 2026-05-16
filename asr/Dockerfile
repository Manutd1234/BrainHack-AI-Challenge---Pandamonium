# Dockerfile for the TIL ASR container.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/workspace/.cache/huggingface \
    ASR_ENGINE=whisper \
    MERALION_MODEL_PATH=/workspace/models/MERaLiON-2-3B \
    WHISPER_MODEL_PATH=/workspace/models/whisper-large-v3-turbo \
    WHISPER_BASE_MODEL_PATH=/workspace/models/whisper-large-v3-turbo \
    ASR_USE_DEEPFILTERNET=0 \
    ASR_USE_WHISPER_FALLBACK=1 \
    ASR_MAX_NEW_TOKENS=256

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model ./model/

# Bake MERaLiON only when explicitly using it; trained Whisper is the default.
ARG MERALION_MODEL_ID=MERaLiON/MERaLiON-2-3B
RUN if [ "${ASR_ENGINE}" != "whisper" ]; then \
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MERALION_MODEL_ID}', local_dir='${MERALION_MODEL_PATH}', local_dir_use_symlinks=False, ignore_patterns=['*.md', '*.png', '*.pdf'])"; \
    fi

# If no fine-tuned Whisper checkpoint was copied in, bake the base model.
ARG WHISPER_MODEL_ID=openai/whisper-large-v3-turbo
RUN if [ ! -f "${WHISPER_MODEL_PATH}/config.json" ]; then \
        python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='${WHISPER_MODEL_ID}', local_dir=os.environ['WHISPER_BASE_MODEL_PATH'], local_dir_use_symlinks=False, ignore_patterns=['*.md', '*.png', '*.pdf'])"; \
    fi

COPY src .

EXPOSE 5001
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "1"]
