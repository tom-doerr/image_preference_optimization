from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import Any


def _server_url(path: str) -> str:
    base = os.getenv("IMAGE_SERVER_URL") or ""
    if not base:
        raise RuntimeError("IMAGE_SERVER_URL not set")
    return base.rstrip("/") + path


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # nosec - URL comes from trusted config
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _image_from_b64_png(b64: str) -> Any:
    try:
        from PIL import Image  # type: ignore

        buf = io.BytesIO(base64.b64decode(b64))
        return Image.open(buf)
    except Exception:
        # Fallback: return raw bytes for tests that don't need real images
        return base64.b64decode(b64)


def generate_image(
    prompt: str, width: int, height: int, steps: int, guidance: float
) -> Any:
    payload = {
        "prompt": prompt,
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "guidance": float(guidance),
    }
    out = _post_json(_server_url("/generate"), payload)
    if not isinstance(out, dict) or "image" not in out:
        raise RuntimeError("image_server: invalid response for /generate")
    return _image_from_b64_png(out["image"])  # expects base64 PNG


def generate_image_latents(
    prompt: str, latents, width: int, height: int, steps: int, guidance: float
) -> Any:
    # Accept numpy or torch tensors; convert to nested lists for JSON
    try:
        import numpy as np  # type: ignore

        if hasattr(latents, "detach"):
            latents = latents.detach().cpu().numpy()
        arr = np.asarray(latents, dtype=float)
        payload = {
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "guidance": float(guidance),
            "latents": arr.tolist(),
            "latents_shape": list(arr.shape),
        }
    except Exception:
        # Last resort: send as-is; server must interpret it
        payload = {
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "guidance": float(guidance),
            "latents": latents,
        }
    out = _post_json(_server_url("/generate_latents"), payload)
    if not isinstance(out, dict) or "image" not in out:
        raise RuntimeError("image_server: invalid response for /generate_latents")
    return _image_from_b64_png(out["image"])  # expects base64 PNG
