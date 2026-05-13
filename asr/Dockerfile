# Dockerfile for building the ASR image.


# The base image, an example deep learning VM.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV WHISPER_MODEL=models/whisper-large-v3
ENV WHISPER_FALLBACK_MODEL=large-v3
ENV WHISPER_DEVICE_INDEX=0
WORKDIR /workspace

# Install ffmpeg for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper large-v3 into a runtime path copied into the image.
COPY download_models.py .
RUN python download_models.py

# Copies your source files.
COPY models/ ./models/
COPY src .

# Starts your model server.
CMD uvicorn asr_server:app --port 5001 --host 0.0.0.0
