# Railway production image — FastAPI + Possession Estimation (YOLO + OpenCV)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCR_MAX_DIMENSION=2800 \
    OCR_BACKEND=tesseract \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    # Tell ultralytics to store weights inside /app so they survive the layer cache
    YOLO_CONFIG_DIR=/app/.yolo

WORKDIR /app

# System libs:
#   libgl1 + libglib2.0-0  → OpenCV headless
#   libgomp1               → NumPy / XGBoost threading
#   tesseract-ocr          → OCR module
#   ffmpeg + libavcodec    → cv2.VideoCapture / VideoWriter on Railway
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    tesseract-ocr \
    tesseract-ocr-fra \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . ./

# Pre-download the YOLO weights at build time so the first request
# doesn't time out while ultralytics fetches 45 MB from the internet.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" || \
    python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Railway supplies PORT at runtime; 8000 keeps the image runnable locally.
# Workers=1 keeps memory predictable; YOLO is already multi-threaded internally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120"]
