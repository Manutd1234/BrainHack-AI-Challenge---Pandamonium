FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 5003

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "5003"]
