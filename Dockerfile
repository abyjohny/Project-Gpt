FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HOST=0.0.0.0

# Install essential build packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY . .

# Ensure persistent data directories exist
RUN mkdir -p data/cv data/jobs data/documents

EXPOSE 8080

CMD ["python", "server.py"]
