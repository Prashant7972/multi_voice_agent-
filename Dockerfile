FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/models /app/audio /app/credentials /app/private

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
