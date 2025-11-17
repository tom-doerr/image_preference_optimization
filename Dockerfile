FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install PyTorch with CUDA 12.1 wheels, then project deps
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip install -r requirements.txt && \
    pip install transformers accelerate pillow

COPY . /app

ENV FLUX_LOCAL_MODEL=stabilityai/sd-turbo \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

