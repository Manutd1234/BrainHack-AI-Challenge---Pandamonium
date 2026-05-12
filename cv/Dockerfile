FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY til26.yaml .

EXPOSE 5002

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "5002"]
