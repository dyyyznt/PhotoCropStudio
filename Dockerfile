FROM python:3.10-slim

# Install system dependencies yang dibutuhkan OpenCV & MediaPipe di Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Salin dan instal dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Salin seluruh file proyek ke dalam container
COPY . .

# Jalankan aplikasi FastAPI dengan port dinamis
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"