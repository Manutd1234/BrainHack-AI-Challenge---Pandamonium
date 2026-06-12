FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TORCH_NUM_THREADS=2 \
    AE_DEVICE=cpu \
    AE_MODEL_PATH=/workspace/models/policy.zip \
    AE_CHECKPOINT_PATH=/workspace/models/policy.zip \
    AE_PICKUP_OVERRIDE=1 \
    AE_ESCAPE_OVERRIDE=1 \
    AE_STUCK_OVERRIDE=1 \
    AE_BOMB_OVERRIDE=1 \
    AE_BLOCK_SPEC_BOMBS=0

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src .
COPY src ./src
COPY models ./models

EXPOSE 5005

CMD ["uvicorn", "ae_server:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]
