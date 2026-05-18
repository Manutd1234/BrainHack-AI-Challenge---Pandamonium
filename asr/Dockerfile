FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    ASR_MODEL_NAME=nvidia/parakeet-tdt-1.1b \
    ASR_MODEL_CACHE=/app/model/parakeet \
    ASR_BATCH_SIZE=8 \
    ASR_USE_WHISPER_FALLBACK=1 \
    ASR_WHISPER_MODE=rescue \
    ASR_WHISPER_MODEL=openai/whisper-large-v3-turbo \
    ASR_WHISPER_CACHE=/app/model/whisper-large-v3-turbo \
    ASR_WHISPER_LANGUAGE=en \
    ASR_WHISPER_MAX_NEW_TOKENS=256 \
    ASR_USE_MEMORY=0 \
    ASR_USE_DOMAIN_CORRECTION=0 \
    ASR_USE_DEEPFILTERNET=0 \
    ASR_DF_MODE=off \
    ASR_MAX_SECONDS=35 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        cargo \
        ffmpeg \
        git \
        libsndfile1 \
        pkg-config \
        rustc \
        sox \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Bake Parakeet-TDT-1.1B into the image so runtime has zero NGC download latency.
RUN python3 -c "import os; import nemo.collections.asr as nemo_asr; os.makedirs('/app/model/parakeet', exist_ok=True); model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-1.1b'); model.save_to('/app/model/parakeet/parakeet-tdt-1.1b.nemo'); print('Parakeet-TDT-1.1B saved')"

# Bake Whisper Large-v3-Turbo for optional rescue inference. It is only used
# when ASR_USE_WHISPER_FALLBACK=1 and Parakeet output looks suspicious.
RUN python3 -c "import os; from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; cache=os.environ.get('ASR_WHISPER_CACHE', '/app/model/whisper-large-v3-turbo'); model_id=os.environ.get('ASR_WHISPER_MODEL', 'openai/whisper-large-v3-turbo'); os.makedirs(cache, exist_ok=True); AutoProcessor.from_pretrained(model_id).save_pretrained(cache); AutoModelForSpeechSeq2Seq.from_pretrained(model_id, low_cpu_mem_usage=True, use_safetensors=True).save_pretrained(cache); print('Whisper Turbo saved')"

COPY src/ ./src/
WORKDIR /app/src

EXPOSE 5001
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "1"]
