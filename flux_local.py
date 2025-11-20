import os
import math
import threading
import logging
from typing import Optional


PIPE = None  # lazily initialized Diffusers pipeline
CURRENT_MODEL_ID = None
PIPE_LOCK = threading.Lock()
LAST_CALL: dict = {}
PROMPT_CACHE: dict = {}
USE_IMAGE_SERVER = False
IMAGE_SERVER_URL = os.getenv("IMAGE_SERVER_URL")

def use_image_server(on: bool, url: Optional[str] = None) -> None:
    global USE_IMAGE_SERVER, IMAGE_SERVER_URL
    USE_IMAGE_SERVER = bool(on)
    if url is not None:
        IMAGE_SERVER_URL = str(url)
LOGGER = logging.getLogger("ipo")
if not LOGGER.handlers:
    try:
        _h = logging.FileHandler("ipo.debug.log")
        _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s flux_local: %(message)s"))
        LOGGER.addHandler(_h)
        LOGGER.setLevel(logging.INFO)
    except Exception:
        pass


def get_last_call() -> dict:
    return dict(LAST_CALL)


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
        try:
            print(f"[pipe] reuse model id={CURRENT_MODEL_ID!r}")
        except Exception:
            pass
        return PIPE

    mid = model_id or _get_model_id()
    if PIPE is not None and CURRENT_MODEL_ID != mid:
        _free_pipe()
    try:
        print(f"[pipe] loading model id={mid!r}")
    except Exception:
        pass
    import time as _time
    t0 = _time.perf_counter()
    PIPE = DiffusionPipeline.from_pretrained(
        mid,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    ).to("cuda")
    try:
        dt = _time.perf_counter() - t0
        print(f"[pipe] loaded model id={mid!r} in {dt:.3f} s")
    except Exception:
        pass
    # Disable safety checker to avoid blacked-out images; keep it minimal.
    try:
        if hasattr(PIPE, 'safety_checker'):
            PIPE.safety_checker = None  # type: ignore[assignment]
        if hasattr(PIPE, 'requires_safety_checker'):
            try:
                PIPE.register_to_config(requires_safety_checker=False)  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(PIPE, 'feature_extractor'):
            PIPE.feature_extractor = None  # type: ignore[assignment]
    except Exception:
        pass
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
    # clear prompt cache on model switch
    try:
        PROMPT_CACHE.clear()
    except Exception:
        pass
    try:
        LAST_CALL.update({"event": "load_model", "model_id": mid})
        LOGGER.info("loaded model %s", mid)
    except Exception:
        pass
    return PIPE


def _to_cuda_fp16(latents):
    """Return a torch fp16 CUDA tensor for the given latents (np/torch)."""
    import torch  # type: ignore
    TensorType = getattr(torch, 'Tensor', None)
    if TensorType is not None and isinstance(latents, TensorType):
        return latents.to(device='cuda', dtype=torch.float16)
    return torch.tensor(latents, dtype=torch.float16, device='cuda')


def _normalize_to_init_sigma(pipe, latents, steps: Optional[int] = None):
    """Scale latents so std ≈ scheduler.init_noise_sigma for current steps.

    We explicitly set timesteps when provided to align with the pipeline's
    schedule, which has proven to avoid degenerate (near-black) outputs on
    some schedulers.
    """
    init_sigma = 1.0
    try:
        sched = getattr(pipe, 'scheduler', None)
        if sched is not None:
            if isinstance(steps, int) and steps > 0 and hasattr(sched, 'set_timesteps'):
                try:
                    sched.set_timesteps(int(steps))
                except Exception:
                    pass
            s = getattr(sched, 'init_noise_sigma', None)
            if s is None:
                sigmas = getattr(sched, 'sigmas', None)
                if sigmas is not None and hasattr(sigmas, 'max'):
                    s = float(sigmas.max().item())
            init_sigma = float(s) if s is not None else 1.0
    except Exception:
        pass
    # Support numpy-backed test stubs that attach `.arr`
    try:
        std = float(latents.std().item()) if latents.numel() else 1.0
    except Exception:
        try:
            import numpy as _np  # type: ignore
            arr = getattr(latents, 'arr', None)
            std = float(_np.asarray(arr).std()) if arr is not None else 1.0
        except Exception:
            std = 1.0
    if not math.isfinite(std) or std == 0.0:
        std = 1.0
    # Arithmetic for stub tensors: mutate in place if possible
    try:
        return (latents / std) * init_sigma
    except Exception:
        arr = getattr(latents, 'arr', None)
        if arr is not None:
            latents.arr = (arr / std) * init_sigma
            return latents
        return latents


def _run_pipe(**kwargs):
    """Call the Diffusers pipeline under the lock and return the first image.

    Also logs basic image statistics for debugging when possible.
    """
    # Best-effort: ensure scheduler has valid timesteps and step index
    try:
        steps = int(kwargs.get("num_inference_steps", 20))
    except Exception:
        steps = 20
    try:
        sched = getattr(PIPE, "scheduler", None)
        if sched is not None and hasattr(sched, "set_timesteps"):
            try:
                # LCM requires _step_index and num_inference_steps to be initialized.
                sched.set_timesteps(int(steps), device="cuda")
            except TypeError:
                sched.set_timesteps(int(steps))  # older signatures
            # Guard against buggy schedulers leaving _step_index=None
            if getattr(sched, "_step_index", None) is None:
                try:
                    sched._step_index = 0  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Some schedulers (e.g. newer LCM) require num_inference_steps
            # to be set explicitly; mirror what set_timesteps would do.
            try:
                if getattr(sched, "num_inference_steps", None) is None:
                    sched.num_inference_steps = int(steps)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    import time as _time
    retries = 0
    try:
        retries = int(os.getenv("RETRY_ON_OOM", "0"))
    except Exception:
        retries = 0
    attempt = 0
    while True:
        try:
            t0 = _time.perf_counter()
            try:
                print(f"[pipe] starting PIPE call event={LAST_CALL.get('event')} steps={steps} w={kwargs.get('width')} h={kwargs.get('height')}")
            except Exception:
                pass
            with PIPE_LOCK:
                out = PIPE(**kwargs)
            dur_s = _time.perf_counter() - t0
            try:
                LAST_CALL["dur_s"] = float(dur_s)
                print(f"[perf] PIPE call took {dur_s:.3f} s (steps={kwargs.get('num_inference_steps')}, w={kwargs.get('width')}, h={kwargs.get('height')})")
            except Exception:
                pass
            if hasattr(out, "images") and getattr(out, "images"):
                img0 = out.images[0]
                try:
                    import numpy as _np  # type: ignore
                    arr = _np.asarray(img0)
                    LAST_CALL.update({
                        "img0_mean": float(arr.mean()),
                        "img0_std": float(arr.std()),
                        "img0_min": float(arr.min()),
                        "img0_max": float(arr.max()),
                    })
                    LOGGER.info(
                        "img0 stats mean=%.3f std=%.3f min=%s max=%s",
                        float(arr.mean()),
                        float(arr.std()),
                        arr.min(),
                        arr.max(),
                    )
                except Exception:
                    pass
                return img0
            raise RuntimeError("Local FLUX pipeline returned no images")
        except RuntimeError as e:
            if attempt < retries and "out of memory" in str(e).lower():
                attempt += 1
                try:
                    import torch  # type: ignore
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                _time.sleep(1.0)
                continue
            raise


def _get_prompt_embeds(prompt: str, guidance: float):
    """Encode prompt once per model/prompt/guidance-mode and cache embeddings.

    For sd‑turbo we use the pipeline's encode_prompt to avoid re‑tokenizing
    and re‑encoding on every rerun. If encode_prompt isn't available, return
    (None, None) and fall back to string prompts in the caller.
    """
    key = (CURRENT_MODEL_ID, prompt, bool(guidance and guidance > 1e-6))
    try:
        if key in PROMPT_CACHE:
            return PROMPT_CACHE[key]
        enc = getattr(PIPE, 'encode_prompt', None)
        if enc is None:
            return (None, None)
        # num_images_per_prompt=1, classifier-free guidance as requested
        prompt_embeds, neg_embeds = enc(
            prompt=prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=bool(key[2]),
            negative_prompt=None,
        )
        PROMPT_CACHE[key] = (prompt_embeds, neg_embeds)
        return PROMPT_CACHE[key]
    except Exception:
        return (None, None)


def generate_flux_image(prompt: str,
                        seed: Optional[int] = None,
                        width: int = 768,
                        height: int = 768,
                        steps: int = 20,
                        guidance: float = 3.5):
    if USE_IMAGE_SERVER and IMAGE_SERVER_URL:
        import image_server as _srv
        LAST_CALL.update({"event": "text_call", "width": int(width), "height": int(height), "steps": int(steps), "guidance": float(guidance)})
        return _srv.generate_image(prompt, int(width), int(height), int(steps), float(guidance))
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

    LAST_CALL.update({
        "event": "text_call",
        "model_id": CURRENT_MODEL_ID,
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "guidance": float(guidance),
    })
    pe, ne = _get_prompt_embeds(prompt, guidance)
    if pe is not None:
        return _run_pipe(
            prompt_embeds=pe,
            negative_prompt_embeds=ne,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=gen,
        )
    else:
        return _run_pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=gen,
        )
    # unreachable; keep signature symmetry


def generate_flux_image_latents(prompt: str,
                                latents,
                                width: int = 768,
                                height: int = 768,
                                steps: int = 20,
                                guidance: float = 3.5):
    if USE_IMAGE_SERVER and IMAGE_SERVER_URL:
        import image_server as _srv
        LAST_CALL.update({"event": "latents_call", "width": int(width), "height": int(height), "steps": int(steps), "guidance": float(guidance)})
        return _srv.generate_image_latents(prompt, latents, int(width), int(height), int(steps), float(guidance))
    # Ensure pipeline is ready (env model id if not set)
    _ensure_pipe(None)

    # Convert and normalize latents succinctly
    latents = _to_cuda_fp16(latents)
    try:
        latents = _normalize_to_init_sigma(PIPE, latents, steps=int(steps))
    except TypeError:
        # test stubs may provide a 2-arg variant
        latents = _normalize_to_init_sigma(PIPE, latents)
    # record basic stats for debugging
    try:
        std = float(latents.std().item()) if hasattr(latents, 'std') else None
    except Exception:
        std = None
    try:
        mean = float(getattr(latents, 'mean')().item()) if hasattr(latents, 'mean') else None
    except Exception:
        mean = None
    try:
        shp = None
        try:
            s = getattr(latents, 'shape', None)
            if s is not None:
                shp = tuple(int(x) for x in s)
        except Exception:
            shp = None
        LAST_CALL.update({
            "event": "latents_call",
            "model_id": CURRENT_MODEL_ID,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "guidance": float(guidance),
            "latents_std": std,
            "latents_mean": mean,
            "latents_shape": shp,
        })
        try:
            sched = getattr(PIPE, 'scheduler', None)
            init_sigma = getattr(sched, 'init_noise_sigma', None)
        except Exception:
            init_sigma = None
        LOGGER.info(
            "latents gen w=%s h=%s steps=%s g=%.3f std=%s mean=%s shape=%s init_sigma=%s",
            width,
            height,
            steps,
            guidance,
            std,
            mean,
            shp,
            init_sigma,
        )
    except Exception:
        pass
    pe, ne = _get_prompt_embeds(prompt, guidance)
    if pe is not None:
        return _run_pipe(
            prompt_embeds=pe,
            negative_prompt_embeds=ne,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            latents=latents,
        )
    else:
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
    pipe = _ensure_pipe(model_id)
    # Use Euler A for SD‑Turbo unless already configured; improves stability
    try:
        if isinstance(model_id, str) and ('sd-turbo' in model_id or 'sdxl-turbo' in model_id):
            # SD‑Turbo models work best with LCM scheduler; EulerA can yield
            # degenerate outputs when injecting latents. Keep it simple.
            from diffusers import LCMScheduler  # type: ignore
            cfg = getattr(pipe, 'scheduler', None).config
            pipe.scheduler = LCMScheduler.from_config(cfg)  # type: ignore[index]
            LOGGER.info("turbo model detected; using LCMScheduler for %s", model_id)
    except Exception:
        pass
