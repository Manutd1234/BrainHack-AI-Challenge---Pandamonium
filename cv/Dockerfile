# Dockerfile for building the CV image.


# The base image, an example deep learning VM.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Install system dependencies for OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the YOLO11l model during build (no internet at runtime).
# Using 'l' (large) variant for best accuracy/speed tradeoff.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"

# Copy fine-tuned model weights if available.
# If you've trained on TIL-26 data, place best.pt in cv/models/
COPY models/ ./models/

# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
