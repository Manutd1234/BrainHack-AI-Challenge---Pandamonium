FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_TF32_OVERRIDE=1 \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

ENV ASR_MODEL_PATH=/workspace/model/parakeet/parakeet_v2_decoder_e1_hard120_replay.nemo \
    ASR_CORRECTIONS_FILE=/workspace/src/asr_corrections.json \
    ASR_VOCABULARY_FILE=/workspace/src/asr_vocabulary.json \
    ASR_USE_MEMORY=0 \
    ASR_MAX_SECONDS=30 \
    ASR_BATCH_SIZE=8 \
    ASR_FP16=1 \
    ASR_DISABLE_CUDA_GRAPHS=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg sox git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scipy

RUN mkdir -p /workspace/model/parakeet
COPY model/parakeet/parakeet_v2_decoder_e1_hard120_replay.nemo /workspace/model/parakeet/parakeet_v2_decoder_e1_hard120_replay.nemo

COPY src /workspace/src
WORKDIR /workspace/src

CMD ["uvicorn", "asr_server:app", "--port", "5001", "--host", "0.0.0.0"]
