# Dockerfile for the TIL Noise container.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    CV_MODEL_PATH=/app/model/best.pt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Reuse the fine-tuned CV checkpoint as the white-box surrogate.
# Copy cv/model/best.pt to noise/model/best.pt before building.
COPY model/best.pt /app/model/best.pt

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY src/ ./src/
WORKDIR /app/src

EXPOSE 5003
CMD ["uvicorn", "noise_server:app", "--host", "0.0.0.0", "--port", "5003", "--workers", "1"]
