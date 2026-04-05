FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p personas knowledge chromadb_data persistent_data

EXPOSE 5555

# Entrypoint: symlink persistent dirs then start gunicorn
CMD ["sh", "-c", "\
  mkdir -p /app/persistent_data/chromadb_data /app/persistent_data/knowledge && \
  rm -rf /app/chromadb_data /app/knowledge && \
  ln -sf /app/persistent_data/chromadb_data /app/chromadb_data && \
  ln -sf /app/persistent_data/knowledge /app/knowledge && \
  gunicorn app:app --bind 0.0.0.0:5555 --workers 2 --threads 4 --timeout 300 \
"]
