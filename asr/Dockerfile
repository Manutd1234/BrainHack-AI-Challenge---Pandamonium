# Dockerfile for the TIL ASR container.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/workspace/.cache/huggingface \
    MERALION_MODEL_PATH=/workspace/models/MERaLiON-2-3B \
    WHISPER_MODEL_PATH=/workspace/models/whisper-small \
    ASR_USE_DEEPFILTERNET=1 \
    ASR_USE_WHISPER_FALLBACK=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake MERaLiON into the image so evaluation can run without network access.
ARG MERALION_MODEL_ID=MERaLiON/MERaLiON-2-3B
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MERALION_MODEL_ID}', local_dir='${MERALION_MODEL_PATH}', local_dir_use_symlinks=False, ignore_patterns=['*.md', '*.png', '*.pdf'])"

# Whisper is a compact fallback for cases where MERaLiON returns an empty string.
ARG WHISPER_MODEL_ID=openai/whisper-small
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${WHISPER_MODEL_ID}', local_dir='${WHISPER_MODEL_PATH}', local_dir_use_symlinks=False, ignore_patterns=['*.md', '*.png', '*.pdf'])"

# Warm the DeepFilterNet3 cache during build instead of first inference.
RUN python -c "from df.enhance import init_df; init_df('DeepFilterNet3', log_level='ERROR', log_file=None)"

COPY src .

EXPOSE 5001
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "1"]
