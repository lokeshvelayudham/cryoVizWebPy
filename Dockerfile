# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps for Pillow (PNG/JPEG) and scipy stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg62-turbo-dev zlib1g-dev libopenjp2-7 libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Switch to user before installing pip dependencies to avoid chown overhead
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Create temp directory for processing
RUN mkdir -p temp

# uvicorn worker
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]