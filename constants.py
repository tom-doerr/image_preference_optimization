from dataclasses import dataclass

APP_VERSION = "0.1.0"
DEFAULT_PROMPT = "neon punk city, women with short hair, standing in the rain"
DEFAULT_MODEL = "stabilityai/sd-turbo"
MODEL_CHOICES = [
    DEFAULT_MODEL,
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
]

# 7 GB VRAM profile clamps
SMALL_VRAM_MAX_WIDTH = 448
SMALL_VRAM_MAX_HEIGHT = 448
SMALL_VRAM_MAX_STEPS = 12

# Minimal config to avoid scattered literals


@dataclass(frozen=True)
class Config:
    DEFAULT_WIDTH: int = 448
    DEFAULT_HEIGHT: int = 448
    DEFAULT_STEPS: int = 6
    DEFAULT_GUIDANCE: float = 3.5

# UI/generation tuning
DECODE_TIMEOUT_S = 3.0

# UI defaults (centralized)
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.5
DEFAULT_TRUST_R = 2.5
DEFAULT_LR_MU = 0.3
DEFAULT_GAMMA_ORTH = 0.2
DEFAULT_ITER_STEPS = 10
DEFAULT_QUEUE_SIZE = 6
DEFAULT_BATCH_SIZE = 25

# Scoring defaults (avoid scattered literals)
DISTANCEHILL_GAMMA = 0.5
COSINEHILL_BETA = 5.0
