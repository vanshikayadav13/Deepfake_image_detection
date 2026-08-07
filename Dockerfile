FROM python:3.11-slim

WORKDIR /app

# System libraries needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ── Step 1: facenet-pytorch WITH all its deps (installs pillow>=10, requests, etc.) ──
RUN pip install --no-cache-dir facenet-pytorch

# ── Step 2: torch + torchvision CPU-only, --no-deps to skip pillow~=9.3 conflict ──
RUN pip install --no-cache-dir \
    torch torchvision \
    --no-deps \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 3: remaining app dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app source ──
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
