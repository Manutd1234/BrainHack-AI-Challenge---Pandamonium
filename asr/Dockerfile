FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights and source code
COPY models/ ./models/
COPY src/ ./src/

EXPOSE 5001

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "5001"]
