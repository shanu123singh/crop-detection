# ============================================================
# Production Dockerfile for Capsicum Crop Detection API
# Optimized for Google Cloud Run (CPU-only PyTorch)
# ============================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install system utilities needed for image processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install PyTorch CPU build first to keep image lightweight (~600MB vs ~4GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py class_info.py ./
COPY models/ ./models/

# Expose port (Cloud Run defaults to 8080)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Launch FastAPI app with Uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
