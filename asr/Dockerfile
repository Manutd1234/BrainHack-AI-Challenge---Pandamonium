FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    ASR_MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2 \
    ASR_MODEL_CACHE=/app/model/parakeet \
    ASR_MODEL_LOAD_MAP_LOCATION=cpu \
    ASR_BATCH_SIZE=16 \
    ASR_USE_AUTOCAST=1 \
    ASR_USE_FP16_WEIGHTS=1 \
    ASR_USE_WHISPER_FALLBACK=0 \
    ASR_WHISPER_MODE=rescue \
    ASR_WHISPER_MODEL=openai/whisper-large-v3-turbo \
    ASR_WHISPER_CACHE=/app/model/whisper-large-v3-turbo \
    ASR_WHISPER_LANGUAGE=en \
    ASR_WHISPER_MAX_NEW_TOKENS=256 \
    ASR_USE_MEMORY=1 \
    ASR_USE_DOMAIN_CORRECTION=1 \
    ASR_USE_DEEPFILTERNET=0 \
    ASR_DF_MODE=off \
    ASR_MAX_SECONDS=35 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ARG ASR_WITH_WHISPER=0

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

# If a decoder-fine-tuned .nemo is present in asr/model/parakeet, COPY it.
# Otherwise bake the smaller Parakeet-TDT-0.6B checkpoint into the image so
# runtime has zero Hugging Face/NGC download latency.
COPY model/ ./model/
RUN python3 -c "import os, re; import nemo.collections.asr as nemo_asr; model_name=os.environ.get('ASR_MODEL_NAME','nvidia/parakeet-tdt-0.6b-v2'); slug=re.sub(r'[^A-Za-z0-9_.-]+','_',model_name.split('/')[-1]); path=os.environ.get('ASR_MODEL_FILE', f'/app/model/parakeet/{slug}.nemo'); os.makedirs(os.path.dirname(path), exist_ok=True); exists=os.path.exists(path) and os.path.getsize(path)>1024; print(f'Using existing Parakeet checkpoint: {path}' if exists else f'Downloading {model_name} to {path}'); model=None if exists else nemo_asr.models.ASRModel.from_pretrained(model_name); None if exists else model.save_to(path); print('Parakeet checkpoint ready')"

# Optional rescue path. Disabled by default because it is large and slower;
# build with --build-arg ASR_WITH_WHISPER=1 only if Parakeet has blank outputs.
RUN if [ "$ASR_WITH_WHISPER" = "1" ]; then python3 -c "import os; from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; cache=os.environ.get('ASR_WHISPER_CACHE', '/app/model/whisper-large-v3-turbo'); model_id=os.environ.get('ASR_WHISPER_MODEL', 'openai/whisper-large-v3-turbo'); os.makedirs(cache, exist_ok=True); AutoProcessor.from_pretrained(model_id).save_pretrained(cache); AutoModelForSpeechSeq2Seq.from_pretrained(model_id, low_cpu_mem_usage=True, use_safetensors=True).save_pretrained(cache); print('Whisper Turbo saved')"; else echo "Skipping Whisper fallback download"; fi

COPY src/ ./src/
WORKDIR /app/src

EXPOSE 5001
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "1"]
