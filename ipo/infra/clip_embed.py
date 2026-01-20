"""SigLIP image embedding."""
import numpy as np

_MODEL, _PROCESSOR = None, None
MODEL_ID = "google/siglip-base-patch16-224"


def load_siglip():
    """Load SigLIP model (cached)."""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import AutoProcessor, AutoModel
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
        _MODEL = AutoModel.from_pretrained(MODEL_ID).to("cuda").eval()
    return _MODEL, _PROCESSOR


def embed_image(img) -> np.ndarray:
    """Embed PIL image to (768,) vector."""
    import torch
    model, proc = load_siglip()
    inputs = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    return out[0].cpu().numpy()
