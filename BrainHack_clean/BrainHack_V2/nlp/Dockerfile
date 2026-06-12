FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

RUN pip install -U pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src .

CMD uvicorn nlp_server:app --port 5004 --host 0.0.0.0