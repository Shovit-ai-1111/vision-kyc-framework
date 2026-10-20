FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app     OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/matplotlib     HF_HOME=/models/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends     tesseract-ocr tesseract-ocr-eng tesseract-ocr-nep libgl1 libglib2.0-0 libgomp1 ca-certificates     && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt     && pip install --no-deps facenet-pytorch==2.6.0
# facenet-pytorch's package metadata pins older torch. We retain its model code,
# explicitly use modern CPU torch, and verify actual inference in model-setup.
COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data
COPY weights/ner-crf /opt/seed/ner-crf
COPY pyproject.toml .
RUN useradd --uid 10001 --create-home appuser && mkdir -p /models /data /evaluations && chown -R appuser:appuser /models /data /evaluations /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
