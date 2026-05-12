FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Pass through OpenAI API Key if needed
# ENV OPENAI_API_KEY=your_key_here

EXPOSE 5004

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "5004"]
