"""FastAPI generation server for image preference optimization."""
import base64
import io
import os
import time
import asyncio
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Generation Server")

# Global state
_pipe = None
_current_model = None
_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    embedding: Optional[list[float]] = None
    embedding_type: str = "pooled"
    width: int = 512
    height: int = 512
    steps: int = 4
    guidance: float = 0.0
    seed: int = 42
    delta_scale: float = 0.1


class BatchRequest(BaseModel):
    prompt: str
    embeddings: list[list[float]]
    embedding_type: str = "pooled"
    width: int = 512
    height: int = 512
    steps: int = 4
    guidance: float = 0.0
    seed: int = 42
    delta_scale: float = 0.1


class ModelRequest(BaseModel):
    model_id: str


def _img_to_base64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _get_default_model() -> str:
    return os.getenv("GEN_SERVER_MODEL", "stabilityai/sd-turbo")


async def _ensure_model(model_id: Optional[str] = None):
    global _pipe, _current_model
    mid = model_id or _current_model or _get_default_model()
    async with _lock:
        if _pipe is not None and _current_model == mid:
            return
        from ipo.infra.flux_pipeline import load_pipeline
        _pipe = load_pipeline(mid)
        _current_model = mid


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": _pipe is not None, "current_model": _current_model}


@app.get("/model")
async def get_model():
    return {"model_id": _current_model, "loaded": _pipe is not None}


@app.get("/models")
async def list_models():
    from ipo.infra.flux_pipeline import SUPPORTED_MODELS
    return {"models": list(SUPPORTED_MODELS.keys())}


@app.post("/model")
async def set_model(req: ModelRequest):
    await _ensure_model(req.model_id)
    return {"status": "ok", "model_id": _current_model}


@app.post("/generate")
async def generate(req: GenerateRequest):
    await _ensure_model()
    t0 = time.time()
    from ipo.infra.flux_pipeline import generate_image
    img = generate_image(
        _pipe, req.prompt, req.embedding, req.embedding_type,
        req.width, req.height, req.steps, req.guidance, req.seed, req.delta_scale)
    dt = (time.time() - t0) * 1000
    return {"image_base64": _img_to_base64(img), "model": _current_model, "time_ms": dt}


@app.post("/batch")
async def batch(req: BatchRequest):
    await _ensure_model()
    t0 = time.time()
    from ipo.infra.flux_pipeline import generate_batch
    imgs = generate_batch(
        _pipe, req.prompt, req.embeddings, req.embedding_type,
        req.width, req.height, req.steps, req.guidance, req.seed, req.delta_scale)
    dt = (time.time() - t0) * 1000
    return {"images": [{"image_base64": _img_to_base64(i)} for i in imgs], "time_ms": dt}
