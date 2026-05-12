FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 5005

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "5005"]
