# Railway production image — FastAPI + Possession Estimation (YOLO nano + OpenCV)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCR_MAX_DIMENSION=2800 \
    OCR_BACKEND=tesseract \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    # Possession engine — nano by default (~200 MB peak vs 1.5 GB for medium)
    # Override on Railway: set YOLO_WEIGHTS=yolo11m.pt for better accuracy on Pro plan
    YOLO_WEIGHTS=yolo11n.pt \
    YOLO_HALF=1 \
    YOLO_CONFIG_DIR=/app/.yolo

WORKDIR /app

# System libs:
#   libgl1 + libglib2.0-0  → OpenCV headless
#   libgomp1               → NumPy/XGBoost threading
#   libsm6 + libxext6      → OpenCV runtime
#   tesseract-ocr          → OCR module
#   ffmpeg                 → cv2.VideoCapture / VideoWriter codec support
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

# Pre-bake YOLO nano weights at BUILD TIME so the first HTTP request
# doesn't trigger a 45 MB download and time out on Railway.
# Nano uses ~200 MB RAM vs ~1.5 GB for medium — safe on any Railway plan.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" || \
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Railway injects PORT at runtime; 8000 is used locally.
# Single worker keeps memory predictable; YOLO is multi-threaded internally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120"]
