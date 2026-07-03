FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache

# 系统依赖：Tesseract OCR + 中文字体 + PyMuPDF/GLIBC
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-chi-sim \
      tesseract-ocr-chi-tra \
      libgl1 \
      libglib2.0-0 \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（充分利用 Docker 层缓存）
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 拷贝源码
COPY app ./app

# 默认配置 FlashRank 模型（轻量，~120MB）
# 如需启用 BAAI/bge-reranker-v2-m3（568MB），设 RERANKER_PROVIDER=local 后取消下行注释
# RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

EXPOSE 8001

# 启动命令；python app/main.py 也可（main.py 自带 __main__ 块）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
