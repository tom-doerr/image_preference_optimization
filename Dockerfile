FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

COPY requirements.txt ./

# Install PyTorch with CUDA 13.0 wheels (for GB10), then project deps
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio --break-system-packages && \
    pip install -r requirements.txt --break-system-packages && \
    pip install transformers accelerate pillow scikit-learn --break-system-packages

COPY . /app

ENV FLUX_LOCAL_MODEL=stabilityai/sd-turbo \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8577

CMD ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=8577", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

