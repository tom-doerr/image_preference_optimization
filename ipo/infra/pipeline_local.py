import os, threading
PIPE = None
CURRENT_MODEL_ID = None
PIPE_LOCK = threading.RLock()
PROMPT_CACHE: dict = {}

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

def _normalize_to_init_sigma(pipe, latents, steps=None):
    # Skip normalization - latents already have std~1
    return latents


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


def _load_pipeline(mid: str):
    """Load Diffusers pipeline with the small set of guarded fallbacks we support.

    Kept separate to reduce branching in _ensure_pipe without changing behavior.
    """
    try:
        import torch  # type: ignore
        from diffusers import DiffusionPipeline  # type: ignore
    except Exception as e:
        raise ValueError("torch/diffusers not installed") from e
    try:
        pipe = DiffusionPipeline.from_pretrained(
            mid,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        ).to("cuda")
    except NotImplementedError:
        try:
            pipe = DiffusionPipeline.from_pretrained(
                mid,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cuda",
            )
        except (NotImplementedError, ValueError) as e2:
            if "meta tensor" not in str(e2) and "device_map" not in str(e2):
                raise
            pipe = DiffusionPipeline.from_pretrained(
                mid,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to("cuda")
    return pipe


def _disable_safety(pipe) -> None:
    try:
        if hasattr(pipe, "safety_checker"): pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"): pipe.feature_extractor = None
    except Exception: pass


def _ensure_pipe(model_id=None):
    global PIPE, CURRENT_MODEL_ID
    import torch
    if not torch.cuda.is_available(): raise ValueError("CUDA not available")

    with PIPE_LOCK:
        if PIPE and (model_id is None or CURRENT_MODEL_ID == model_id): return PIPE
    mid = model_id or CURRENT_MODEL_ID or _get_default_model_id()
    if PIPE and CURRENT_MODEL_ID != mid: _free_pipe()
    PIPE = _load_pipeline(mid)
    _disable_safety(PIPE); _post_load_toggles(PIPE)
    CURRENT_MODEL_ID = mid; PROMPT_CACHE.clear()
    return PIPE


def _prepare_scheduler_locked(steps):
    try:
        sched = getattr(PIPE, "scheduler", None)
        if sched and hasattr(sched, "set_timesteps"):
            try: sched.set_timesteps(int(steps), device="cuda")
            except TypeError: sched.set_timesteps(int(steps))
    except Exception: pass


def _run_pipe(**kwargs):
    steps = int(kwargs.get("num_inference_steps", 20))
    with PIPE_LOCK:
        _prepare_scheduler_locked(steps)
        out = PIPE(**kwargs)
    if hasattr(out, "images") and out.images:
        img = out.images[0]
        import numpy as np
        arr = np.array(img)
        print(f"[out] img mean={arr.mean():.1f} std={arr.std():.1f}")
        return img
    raise RuntimeError("Pipeline returned no images")


def _post_load_toggles(pipe):
    try:
        if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"): pipe.enable_vae_slicing()
    except Exception: pass


def _get_prompt_embeds(prompt, guidance):
    if not guidance or guidance <= 1e-6: return (None, None)
    key = (CURRENT_MODEL_ID, prompt)
    if key in PROMPT_CACHE: return PROMPT_CACHE[key]
    try:
        enc = getattr(PIPE, "encode_prompt", None)
        if not enc: return (None, None)
        pe, ne = enc(prompt=prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None)  # noqa: E501
        PROMPT_CACHE[key] = (pe, ne); return pe, ne
    except Exception: return (None, None)


def generate_flux_image_latents(prompt, latents, width=768, height=768, steps=20, guidance=3.5):
    _ensure_pipe(None)
    latents = _to_cuda_fp16(latents)
    # Scale latents by scheduler init_noise_sigma
    sigma = float(getattr(PIPE.scheduler, "init_noise_sigma", 1.0))
    latents = latents * sigma
    g = _eff_g(CURRENT_MODEL_ID or "", guidance)
    print(f"[gen] sigma={sigma:.2f} std={latents.std().item():.3f} g={g} steps={steps}")
    h, w = latents.shape[2] * 8, latents.shape[3] * 8
    kw = dict(num_inference_steps=int(steps or 4), guidance_scale=float(g), latents=latents, prompt=prompt, height=h, width=w)
    return _run_pipe(**kw)




def set_model(model_id):
    global CURRENT_MODEL_ID
    with PIPE_LOCK: pipe = _ensure_pipe(model_id)
    mid = CURRENT_MODEL_ID or ""
    try:
        if "sd-turbo" in mid or "sdxl-turbo" in mid:
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            print(f"[set_model] LCMScheduler set for {mid}")
    except Exception: pass
