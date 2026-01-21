import os
import threading

from ipo.infra.model_registry import resolve_model_id as _resolve_model_id

PIPE = None
CURRENT_MODEL_ID = None
PIPE_LOCK = threading.RLock()
_LCM_SET = False

def _get_default_model_id() -> str:
    return os.getenv("FLUX_LOCAL_MODEL") or "stabilityai/sd-turbo"

def _eff_g(mid: str, g: float) -> float:
    return 0.0 if isinstance(mid, str) and ("sd-turbo" in mid or "sdxl-turbo" in mid) else g
def _to_cuda_fp16(latents):
    try:
        import torch
        if isinstance(latents, torch.Tensor):
            return latents.to(device="cuda", dtype=torch.float16)
        return torch.tensor(latents, dtype=torch.float16, device="cuda")
    except Exception:
        return latents

def _free_pipe() -> None:
    """Best-effort free of cached pipeline and CUDA memory."""
    global PIPE, _LCM_SET
    try:
        import gc  # type: ignore

        import torch  # type: ignore

        if PIPE is not None:
            del PIPE
            PIPE = None
            _LCM_SET = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[_free_pipe] cleanup failed: {e}")


def _load_flux_nf4(mid):
    from ipo.infra.flux_pipeline import _load_by_config
    from ipo.infra.model_registry import MODELS
    cfg = next((v for v in MODELS.values() if v["hf_id"] == mid), None)
    return _load_by_config(cfg)


def _load_pipeline(mid: str):
    """Load Diffusers pipeline, with NF4 quantization for Flux models."""
    try:
        import torch
        from diffusers import DiffusionPipeline, FluxPipeline
    except Exception as e:
        raise ValueError("torch/diffusers not installed") from e

    from ipo.infra.model_registry import MODELS
    cfg = next((v for v in MODELS.values() if v["hf_id"] == mid), None)
    is_flux = "flux" in mid.lower()

    if cfg and cfg.get("quantize") and is_flux:
        return _load_flux_nf4(mid)

    dtype = torch.bfloat16 if is_flux else torch.float16
    Pipe = FluxPipeline if is_flux else DiffusionPipeline
    print(f"[load] {mid} pipe={Pipe.__name__}")
    try:
        pipe = Pipe.from_pretrained(mid, torch_dtype=dtype).to("cuda")
    except NotImplementedError:
        try:
            pipe = Pipe.from_pretrained(mid, torch_dtype=dtype, device_map="cuda")
        except (NotImplementedError, ValueError) as e2:
            if "meta tensor" not in str(e2) and "device_map" not in str(e2):
                raise
            pipe = Pipe.from_pretrained(mid, torch_dtype=dtype).to("cuda")
    return pipe


def _disable_safety(pipe) -> None:
    try:
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None
    except Exception as e:
        print(f"[_disable_safety] failed: {e}")


def _ensure_pipe(model_id=None):
    global PIPE, CURRENT_MODEL_ID
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA not available")

    with PIPE_LOCK:
        if PIPE and (model_id is None or CURRENT_MODEL_ID == model_id):
            return PIPE
    mid = model_id or CURRENT_MODEL_ID or _get_default_model_id()
    if PIPE and CURRENT_MODEL_ID != mid:
        _free_pipe()
    PIPE = _load_pipeline(mid)
    _disable_safety(PIPE)
    _post_load_toggles(PIPE)
    CURRENT_MODEL_ID = mid
    return PIPE


def _prepare_scheduler_locked(steps):
    try:
        sched = getattr(PIPE, "scheduler", None)
        if sched and hasattr(sched, "set_timesteps"):
            try:
                sched.set_timesteps(int(steps), device="cuda")
            except TypeError:
                sched.set_timesteps(int(steps))
    except Exception as e:
        print(f"[_prepare_scheduler] failed: {e}")


def _run_pipe(return_all=False, **kwargs):
    steps = int(kwargs.get("num_inference_steps", 20))
    with PIPE_LOCK:
        _prepare_scheduler_locked(steps)
        out = PIPE(**kwargs)
    if hasattr(out, "images") and out.images:
        if return_all:
            return list(out.images)
        img = out.images[0]
        import numpy as np
        arr = np.array(img)
        print(f"[out] img mean={arr.mean():.1f} std={arr.std():.1f}")
        return img
    raise RuntimeError("Pipeline returned no images")


def _post_load_toggles(pipe):
    try:
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
    except Exception as e:
        print(f"[_post_load_toggles] failed: {e}")


def generate_flux_image_latents(prompt, latents, width=768, height=768, steps=20, guidance=3.5):
    _ensure_pipe(None)
    latents = _to_cuda_fp16(latents)
    # Scale latents by scheduler init_noise_sigma
    sigma = float(getattr(PIPE.scheduler, "init_noise_sigma", 1.0))
    latents = latents * sigma
    g = _eff_g(CURRENT_MODEL_ID or "", guidance)
    print(f"[gen] sigma={sigma:.2f} std={latents.std().item():.3f} g={g} steps={steps}")
    h, w = latents.shape[2] * 8, latents.shape[3] * 8
    kw = dict(
        num_inference_steps=int(steps or 4), guidance_scale=float(g),
        latents=latents, prompt=prompt, height=h, width=w
    )
    return _run_pipe(**kw)


def generate(prompt, width=512, height=512, steps=4, guidance=0.0, seed=42):
    """Simple text-to-image generation."""
    import torch
    _ensure_pipe(None)
    g = _eff_g(CURRENT_MODEL_ID or "", guidance)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    return _run_pipe(prompt=prompt, width=width, height=height,
        num_inference_steps=int(steps), guidance_scale=float(g), generator=gen)


def set_model(model_id):
    global CURRENT_MODEL_ID, _LCM_SET
    raw = model_id if model_id is not None else _get_default_model_id()
    target = _resolve_model_id(raw)
    with PIPE_LOCK:
        if PIPE is not None and CURRENT_MODEL_ID == target:
            print(f"[set_model] {target} already loaded, skip")
            return
        pipe = _ensure_pipe(target)
    if _LCM_SET:
        return
    mid = CURRENT_MODEL_ID or ""
    try:
        if "sd-turbo" in mid or "sdxl-turbo" in mid:
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            _LCM_SET = True
            print(f"[set_model] LCMScheduler set for {mid}")
    except Exception as e:
        print(f"[set_model] LCMScheduler setup failed: {e}")




