#!/usr/bin/env bash
set -euo pipefail

# Minimal venv setup for GTX 1080 Ti (CUDA 11.8)
# Usage: scripts/setup_venv.sh [cu118|cu121]

CUDA_TAG="${1:-cu118}"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

case "$CUDA_TAG" in
  cu118)
    # PyTorch with CUDA 11.8 (Pascal-friendly)
    python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
    ;;
  cu121)
    # PyTorch with CUDA 12.1
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
    ;;
  *)
    echo "Unknown CUDA tag: $CUDA_TAG (use cu118 or cu121)" >&2
    exit 1
    ;;
esac

# Rest of deps
python -m pip install -r requirements.txt

echo "Verifying CUDA availability in torch:"
python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY

echo "\nDone. Activate with: source .venv/bin/activate"
