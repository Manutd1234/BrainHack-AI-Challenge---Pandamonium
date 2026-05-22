FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3-pip python3.11-dev git \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Trained checkpoint — run ae_train.py first
COPY model/ /app/model/

ENV AE_CHECKPOINT_PATH=/app/model/policy.zip
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY src/ ./src/
WORKDIR /app/src
EXPOSE 5005
CMD ["uvicorn", "ae_server:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]
