from __future__ import annotations

import numpy as np
from PIL import Image


def image_to_z(img: Image.Image, lstate) -> np.ndarray:
    """Convert an uploaded RGB image to a latent vector matching lstate.d.

    Minimal mapping: resize to (W/8,H/8), pad to 4 channels, zero-mean per channel.
    """
    h8, w8 = lstate.height // 8, lstate.width // 8
    arr = np.asarray(img.convert("RGB").resize((w8, h8)))
    arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
    pad = np.zeros((h8, w8, 1), dtype=np.float32)
    arr = np.concatenate([arr, pad], axis=2)
    arr = arr - arr.mean(axis=(0, 1), keepdims=True)
    lat = arr.transpose(2, 0, 1)  # (4, h8, w8)
    return lat.reshape(-1)

