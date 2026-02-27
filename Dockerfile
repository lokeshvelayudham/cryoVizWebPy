# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps for Pillow (PNG/JPEG) and scipy stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg62-turbo-dev zlib1g-dev libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# uvicorn worker
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]