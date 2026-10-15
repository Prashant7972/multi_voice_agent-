FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Create non-root application user
RUN useradd -m -u 1000 -s /bin/bash appuser

COPY . .

RUN mkdir -p /app/models /app/audio /app/credentials /app/private /app/app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]

