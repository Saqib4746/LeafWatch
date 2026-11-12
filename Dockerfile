FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('underdogquality/yolo11s-pest-detection', 'best.pt', local_dir='/app/model-source')"

COPY serve.py mobile_detector.py train.py ./
COPY dist ./dist

EXPOSE 8000

CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
