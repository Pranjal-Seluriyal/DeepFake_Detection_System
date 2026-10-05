# DeepTrust AI - Production Docker Container
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

WORKDIR /app

# Install system audio, video, and image libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to avoid heavy GPU wheel downloads in cloud container
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Ensure uploads directory exists
RUN mkdir -p uploads

# Expose standard web port
EXPOSE 5000

# Start production server (1 worker + multithreading prevents duplicating 2.44GB model in RAM)
CMD ["gunicorn", "wsgi:app", "--workers", "1", "--threads", "4", "--timeout", "300", "--bind", "0.0.0.0:5000"]
