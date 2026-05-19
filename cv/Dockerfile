FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Download the YOLO26x pretrained backbone during build.
ARG YOLO_BASE_WEIGHTS=yolo26x.pt
RUN python -c "from ultralytics import YOLO; YOLO('${YOLO_BASE_WEIGHTS}')"

# Run cv_train.py first so at least model/best.pt exists before docker build.
# Copy the whole model folder so optional candidates like last.pt can be ensembled.
COPY model/ /app/model/
RUN test -f /app/model/best.pt

ENV CV_MODEL_PATH=/app/model/best.pt \
    CV_MODEL_PATHS=/app/model/best.pt \
    CV_RFDETR_PATH=/app/model/rfdetr_best.pt \
    CV_USE_RFDETR=1 \
    CV_RFDETR_CONF=0.12 \
    CV_RFDETR_CONF_WEIGHT=0.85 \
    CV_RFDETR_EMPTY_ONLY=0 \
    CV_THRESHOLD_CONFIG=/app/src/cv_thresholds.json \
    CV_USE_SAHI=0 \
    CV_SAHI_MIN_SIZE=900 \
    CV_SAHI_SLICE_SIZE=768 \
    CV_SAHI_OVERLAP=0.25 \
    CV_SAHI_NMS_IOU=0.50 \
    CV_MAX_DETECTIONS=12 \
    CV_RAW_MAX_DETECTIONS=50 \
    CV_MODEL_CONF=0.10 \
    CV_PRED_BATCH_SIZE=4 \
    CV_AUGMENT=0 \
    CV_USE_WBF=1 \
    CV_FALLBACK_FLIP=0 \
    CV_HIGHRES_FALLBACK=1 \
    CV_HIGHRES_IMGSZ=1536 \
    CV_HIGHRES_MAX_COUNT=0 \
    CV_HIGHRES_MIN_CONF=0.45 \
    CV_FINAL_NMS_IOU=0.55 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY src/ ./src/
WORKDIR /app/src

EXPOSE 5002
CMD ["uvicorn", "cv_server:app", "--host", "0.0.0.0", "--port", "5002", "--workers", "1"]
