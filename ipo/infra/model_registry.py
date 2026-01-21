"""Unified model registry - single source of truth for model metadata."""
import torch

MODELS = {
    "sd-turbo": {
        "hf_id": "stabilityai/sd-turbo",
        "pipeline": "DiffusionPipeline",
        "dtype": torch.float16,
        "quantize": False,
        "guidance": 0.0,
        "scheduler": "LCM",
        "size": "1.3GB",
        "precision": "FP16",
    },
    "flux-schnell": {
        "hf_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "FluxPipeline",
        "dtype": torch.bfloat16,
        "quantize": True,
        "guidance": 0.0,
        "size": "23GB",
        "precision": "NF4",
    },
    "flux-dev": {
        "hf_id": "black-forest-labs/FLUX.1-dev",
        "pipeline": "FluxPipeline",
        "dtype": torch.bfloat16,
        "quantize": True,
        "guidance": 3.5,
        "size": "23GB",
        "precision": "NF4",
    },
}

MODEL_OPTIONS = list(MODELS.keys())


def resolve_model_id(key_or_id: str) -> str:
    """Resolve short name to HuggingFace ID."""
    if key_or_id in MODELS:
        return MODELS[key_or_id]["hf_id"]
    return key_or_id


def get_model_config(key: str) -> dict:
    """Get full config for a model key."""
    return MODELS.get(key)
