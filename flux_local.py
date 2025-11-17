import os
import math
import threading
from typing import Optional


PIPE = None  # lazily initialized Diffusers pipeline
CURRENT_MODEL_ID = None
PIPE_LOCK = threading.Lock()


def _get_model_id() -> str:
    """Read default model id from env for implicit loads.

    The app usually calls set_model(...), but tests rely on this path.
    """
    mid = os.getenv("FLUX_LOCAL_MODEL")
    if not mid:
        raise ValueError("FLUX_LOCAL_MODEL not set (e.g. 'black-forest-labs/FLUX.1-schnell')")
    return mid


def _free_pipe() -> None:
    """Best-effort free of cached pipeline and CUDA memory."""
    global PIPE
    try:
        import gc  # type: ignore
        import torch  # type: ignore
        if PIPE is not None:
            del PIPE
            PIPE = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # type: ignore[attr-defined]
    except Exception:
        pass


def _ensure_pipe(model_id: Optional[str] = None):
    """Ensure a CUDA pipeline is loaded for the given or env model id."""
    global PIPE, CURRENT_MODEL_ID
    try:
        import torch  # type: ignore
        from diffusers import DiffusionPipeline  # type: ignore
    except Exception as e:
        raise ValueError("torch/diffusers not installed") from e
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU not available; require 1080 Ti (cuda)")

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    # Return cached if suitable
    if PIPE is not None and (model_id is None or CURRENT_MODEL_ID == model_id):
        return PIPE

    mid = model_id or _get_model_id()
    if PIPE is not None and CURRENT_MODEL_ID != mid:
        _free_pipe()
    PIPE = DiffusionPipeline.from_pretrained(
        mid,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    ).to("cuda")
    try:
        if hasattr(PIPE, 'enable_attention_slicing'):
            PIPE.enable_attention_slicing()
        if hasattr(PIPE, 'enable_vae_slicing'):
            PIPE.enable_vae_slicing()
        if hasattr(PIPE, 'enable_xformers_memory_efficient_attention'):
            PIPE.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    CURRENT_MODEL_ID = mid
    return PIPE


def _to_cuda_fp16(latents):
    """Return a torch fp16 CUDA tensor for the given latents (np/torch)."""
    import torch  # type: ignore
    TensorType = getattr(torch, 'Tensor', None)
    if TensorType is not None and isinstance(latents, TensorType):
        return latents.to(device='cuda', dtype=torch.float16)
    return torch.tensor(latents, dtype=torch.float16, device='cuda')


def _normalize_to_init_sigma(pipe, latents):
    """Scale latents so std ≈ scheduler init sigma; tolerate missing attrs."""
    init_sigma = 1.0
    try:
        sched = getattr(pipe, 'scheduler', None)
        if sched is not None:
            s = getattr(sched, 'init_noise_sigma', None)
            if s is None:
                sigmas = getattr(sched, 'sigmas', None)
                if sigmas is not None and hasattr(sigmas, 'max'):
                    s = float(sigmas.max().item())
            init_sigma = float(s) if s is not None else 1.0
    except Exception:
        pass
    std = float(latents.std().item()) if latents.numel() else 1.0
    if not math.isfinite(std) or std == 0.0:
        std = 1.0
    return (latents / std) * init_sigma


def _run_pipe(**kwargs):
    """Call the Diffusers pipeline under the lock and return the first image."""
    with PIPE_LOCK:
        out = PIPE(**kwargs)
    if hasattr(out, "images") and getattr(out, "images"):
        return out.images[0]
    raise RuntimeError("Local FLUX pipeline returned no images")


def generate_flux_image(prompt: str,
                        seed: Optional[int] = None,
                        width: int = 768,
                        height: int = 768,
                        steps: int = 20,
                        guidance: float = 3.5):
    """Generate one image with a local FLUX model via Diffusers.

    Strict requirements (no fallbacks):
    - torch with CUDA available (1080 Ti)
    - diffusers installed
    - env FLUX_LOCAL_MODEL set to a valid HF model id
    Returns a PIL image (whatever the pipeline returns at out.images[0]).
    """
    # Ensure pipeline is ready (env model id if not set)
    _ensure_pipe(None)

    gen = None
    if seed is not None:
        import torch  # local import for type
        gen = torch.Generator(device="cuda").manual_seed(int(seed))

    return _run_pipe(
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        generator=gen,
    )


def generate_flux_image_latents(prompt: str,
                                latents,
                                width: int = 768,
                                height: int = 768,
                                steps: int = 20,
                                guidance: float = 3.5):
    # Ensure pipeline is ready (env model id if not set)
    _ensure_pipe(None)

    # Convert and normalize latents succinctly
    latents = _to_cuda_fp16(latents)
    latents = _normalize_to_init_sigma(PIPE, latents)
    return _run_pipe(
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        latents=latents,
    )
def set_model(model_id: str):
    """Load or switch the local FLUX/SD/SDXL model. CUDA only; no fallbacks."""
    _ensure_pipe(model_id)
