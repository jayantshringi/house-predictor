FROM python:3.11-slim

WORKDIR /app

# System dependencies layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source system directories
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 5000

# Execute server using multi-threaded worker configurations
CMD ["gunicorn", "--workers=4", "--threads=2", "--bind", "0.0.0.0:5000", "src.app:app"]
