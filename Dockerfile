FROM python:3.11-slim

# Build deps for chromadb / psycopg / bcrypt. Kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Data dir for Chroma + uploads (mounted as a volume in compose)
RUN mkdir -p /app/data/chroma /app/data/uploads

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
