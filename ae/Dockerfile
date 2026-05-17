# Dockerfile for the TIL AE container.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    AE_CHECKPOINT_PATH=/app/model/policy.zip \
    AE_POLICY_MODE=hybrid

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Optional trained checkpoint. The image still runs with only .gitkeep present.
COPY model/ ./model/

COPY src/ ./src/
WORKDIR /app/src

EXPOSE 5005
CMD ["uvicorn", "ae_server:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]
