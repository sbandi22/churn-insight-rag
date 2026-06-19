FROM python:3.11-slim

WORKDIR /app

# Build deps for sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the vector index at image build time so the demo is instant.
RUN python -m ingestion.build_index || true

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
