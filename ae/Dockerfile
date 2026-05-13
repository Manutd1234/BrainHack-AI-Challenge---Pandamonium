# Dockerfile for building the AE image.

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY src .

CMD uvicorn ae_server:app --port 5005 --host 0.0.0.0
