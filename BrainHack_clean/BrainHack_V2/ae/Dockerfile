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
    AE_CHECKPOINT_PATH=/app/model/policy.zip

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY model ./model
COPY src ./src

WORKDIR /app/src

EXPOSE 5005

CMD ["uvicorn", "ae_server:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]
