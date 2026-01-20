"""Unified pipeline loader for SD and Flux models with quantization support."""
import torch

from ipo.infra.model_registry import MODELS

# Alias for backward compatibility
SUPPORTED_MODELS = MODELS


def _get_bnb_config():
    from diffusers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)


def load_pipeline(model_key: str):
    """Load pipeline by model key."""
    if model_key not in SUPPORTED_MODELS:
        for k, v in SUPPORTED_MODELS.items():
            if v["hf_id"] == model_key:
                model_key = k
                break
        else:
            raise ValueError(f"Unknown model: {model_key}")
    return _load_by_config(SUPPORTED_MODELS[model_key])


def _load_by_config(cfg):
    """Load pipeline from config dict."""
    hf_id = cfg["hf_id"]
    dtype = cfg["dtype"]
    quantize = cfg.get("quantize", False)
    pipe_type = cfg["pipeline"]

    if pipe_type == "FluxPipeline":
        from diffusers import FluxPipeline, FluxTransformer2DModel
        if quantize:
            transformer = FluxTransformer2DModel.from_pretrained(
                hf_id, subfolder="transformer", quantization_config=_get_bnb_config(), torch_dtype=dtype)
            pipe = FluxPipeline.from_pretrained(hf_id, transformer=transformer, torch_dtype=dtype)
        else:
            pipe = FluxPipeline.from_pretrained(hf_id, torch_dtype=dtype).to("cuda")
    else:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(hf_id, torch_dtype=dtype, low_cpu_mem_usage=False).to("cuda")

    # Apply LCM scheduler for turbo models
    if cfg.get("scheduler") == "LCM":
        from diffusers import LCMScheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    _disable_safety(pipe)
    return pipe


def _disable_safety(pipe):
    try:
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None
    except Exception:
        pass


def generate_image(pipe, prompt, embedding, emb_type, w, h, steps, guidance, seed, scale):
    """Generate single image."""
    import numpy as np
    gen = torch.Generator("cuda").manual_seed(int(seed))
    # For now, use prompt directly (embedding support for Flux needs text_encoder)
    out = pipe(prompt=prompt, width=w, height=h, num_inference_steps=steps, guidance_scale=guidance, generator=gen)
    return out.images[0]


def generate_batch(pipe, prompt, embeddings, emb_type, w, h, steps, guidance, seed, scale):
    """Generate batch of images."""
    gen = torch.Generator("cuda").manual_seed(int(seed))
    n = len(embeddings)
    out = pipe(prompt=[prompt]*n, width=w, height=h, num_inference_steps=steps, guidance_scale=guidance, generator=gen)
    return list(out.images)
