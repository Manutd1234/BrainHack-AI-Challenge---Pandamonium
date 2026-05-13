# Dockerfile for building the adversarial noise image.

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV YOLO_SURROGATE_FALLBACK=yolo26l.pt
WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_models.py .
RUN python download_models.py --model "${YOLO_SURROGATE_FALLBACK}"

COPY models/ ./models/
COPY src .

CMD uvicorn noise_server:app --port 5003 --host 0.0.0.0
