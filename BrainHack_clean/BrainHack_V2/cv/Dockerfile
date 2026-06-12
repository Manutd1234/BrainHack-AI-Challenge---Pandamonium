FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    CV_MODEL_PATH=/workspace/model/yolo_best_refine15.pt \
    CV_IMGSZ=1280 \
    CV_CONF=0.70 \
    CV_IOU=0.45 \
    CV_MAX_DET=100

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model ./model
COPY src ./src

WORKDIR /workspace/src

CMD ["uvicorn", "cv_server:app", "--port", "5002", "--host", "0.0.0.0", "--workers", "1"]
