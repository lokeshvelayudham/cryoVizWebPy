# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps for Pillow (PNG/JPEG) and scipy stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg62-turbo-dev zlib1g-dev libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set permissions for HF to write temporary TIFF slices
RUN chown -R user:user /app
USER user

# uvicorn worker
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]