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
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512
    DEFAULT_STEPS: int = 6
    DEFAULT_GUIDANCE: float = 3.5

# UI/generation tuning
DECODE_TIMEOUT_S = 3.0
