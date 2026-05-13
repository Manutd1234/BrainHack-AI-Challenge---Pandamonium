# Dockerfile for building the CV image.

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV YOLO_FALLBACK_MODEL=yolo26l.pt
ENV YOLO_IMGSZ=1536
ENV YOLO_CONF=0.10
ENV YOLO_MAX_DET=200
ENV SAHI_SLICE_SIZE=1024
ENV SAHI_OVERLAP=0.30
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_models.py .
RUN python download_models.py --model "${YOLO_FALLBACK_MODEL}"

COPY models/ ./models/
COPY src .

CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
