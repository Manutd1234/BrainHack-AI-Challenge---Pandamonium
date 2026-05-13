# Dockerfile for building the ASR image.


# The base image, an example deep learning VM.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Install ffmpeg for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper large-v3 model during build so it's cached in the image.
# No internet is available at runtime, so all weights must be bundled in.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn asr_server:app --port 5001 --host 0.0.0.0
