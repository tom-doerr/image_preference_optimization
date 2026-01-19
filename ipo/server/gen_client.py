"""Client for generation server."""
import base64
import io
from typing import Optional

import requests
from PIL import Image


class GenerationClient:
    def __init__(self, base_url: str = "http://localhost:8580"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _decode_image(self, b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64)))

    def health(self) -> dict:
        r = self.session.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def get_model(self) -> dict:
        r = self.session.get(f"{self.base_url}/model", timeout=5)
        r.raise_for_status()
        return r.json()

    def set_model(self, model_id: str) -> dict:
        r = self.session.post(f"{self.base_url}/model", json={"model_id": model_id}, timeout=120)
        r.raise_for_status()
        return r.json()

    def generate(self, prompt, emb=None, emb_type="pooled", w=512, h=512, steps=4, g=0.0, seed=42, scale=0.1):
        data = {"prompt": prompt, "embedding": emb, "embedding_type": emb_type,
                "width": w, "height": h, "steps": steps, "guidance": g, "seed": seed, "delta_scale": scale}
        r = self.session.post(f"{self.base_url}/generate", json=data, timeout=120)
        r.raise_for_status()
        return self._decode_image(r.json()["image_base64"])

    def generate_batch(self, prompt, embs, emb_type="pooled", w=512, h=512, steps=4, g=0.0, seed=42, scale=0.1):
        data = {"prompt": prompt, "embeddings": embs, "embedding_type": emb_type,
                "width": w, "height": h, "steps": steps, "guidance": g, "seed": seed, "delta_scale": scale}
        r = self.session.post(f"{self.base_url}/batch", json=data, timeout=300)
        r.raise_for_status()
        return [self._decode_image(x["image_base64"]) for x in r.json()["images"]]
