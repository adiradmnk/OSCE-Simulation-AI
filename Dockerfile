# Gunakan python slim agar image ringan
FROM python:3.11-slim

# Install system dependencies (penting untuk library audio/media jika dipakai)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgobject-2.0-0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app



# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade --upgrade-strategy eager -r requirements.txt

# Copy seluruh kode
COPY . .

EXPOSE 50051

# Jalankan entrypoint agent
# Sesuaikan 'main' dengan nama file main.py kamu
CMD ["python", "main.py"]