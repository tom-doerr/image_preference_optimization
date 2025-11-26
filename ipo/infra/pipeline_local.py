import logging
import os
import threading
from typing import Optional

PIPE = None  # lazily initialized Diffusers pipeline
CURRENT_MODEL_ID = None
PIPE_LOCK = threading.RLock()
LAST_CALL: dict = {}
PROMPT_CACHE: dict = {}
# Image-server support removed for simplicity (local pipeline only)


LOGGER = logging.getLogger("ipo")
if not LOGGER.handlers:
    try:
        _h = logging.FileHandler("ipo.debug.log")
        _h.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s flux_local: %(message)s")
        )
        LOGGER.addHandler(_h)
        LOGGER.setLevel(logging.INFO)
    except Exception:
        pass
try:
    import os as _os

    _lvl = (_os.getenv("IPO_LOG_LEVEL") or "").upper()
    if _lvl:
        LOGGER.setLevel(getattr(logging, _lvl, logging.INFO))
except Exception:
    pass

from .pipeline_utils import (  # noqa: E402
    eff_guidance as _eff_g,
)
from .pipeline_utils import (  # noqa: E402
    get_default_model_id as _get_default_model_id,
)
from .pipeline_utils import (  # noqa: E402
    log_verbosity as _lv,
)
from .pipeline_utils import (  # noqa: E402
    normalize_to_init_sigma as _normalize_to_init_sigma,
)
from .pipeline_utils import (  # noqa: E402
    to_cuda_fp16 as _to_cuda_fp16,
)


def get_last_call() -> dict:
    return dict(LAST_CALL)



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
    """Disable safety components on the pipeline (best-effort)."""
    try:
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None  # type: ignore[assignment]
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None  # type: ignore[assignment]
        try:
            if hasattr(pipe, "register_to_config"):
                pipe.register_to_config(requires_safety_checker=False)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            cfg = getattr(pipe, "config", None)
            if cfg is not None:
                setattr(cfg, "requires_safety_checker", False)
        except Exception:
            pass
        try:
            if _lv() >= 1:
                LOGGER.info("safety checker disabled for model %s", CURRENT_MODEL_ID)
        except Exception:
            pass
    except Exception:
        pass


def _ensure_pipe(model_id: Optional[str] = None):
    """Ensure a CUDA pipeline is loaded for the given or env model id."""
    global PIPE, CURRENT_MODEL_ID
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ValueError("torch not installed") from e
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU not available; require 1080 Ti (cuda)")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Return cached if suitable
    with PIPE_LOCK:
        if PIPE is not None and (model_id is None or CURRENT_MODEL_ID == model_id):
            return PIPE

    # Prefer an explicitly selected/current model id before consulting env
    mid = model_id or CURRENT_MODEL_ID or _get_model_id()
    if PIPE is not None and CURRENT_MODEL_ID != mid:
        _free_pipe()
    PIPE = _load_pipeline(mid)
    _disable_safety(PIPE)
    _post_load_toggles(PIPE)
    CURRENT_MODEL_ID = mid
    _after_model_switch(mid)
    return PIPE


# helpers moved to flux_utils


def _prepare_scheduler_locked(steps: int) -> None:
    """Prepare scheduler timesteps and step_index while PIPE_LOCK is held."""
    try:
        sched = getattr(PIPE, "scheduler", None)
        if sched is None:
            return
        if hasattr(sched, "set_timesteps"):
            try:
                sched.set_timesteps(int(steps), device="cuda")
            except TypeError:
                sched.set_timesteps(int(steps))
        if getattr(sched, "_step_index", None) is None:
            try:
                sched._step_index = 0  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            if getattr(sched, "num_inference_steps", None) is None:
                sched.num_inference_steps = int(steps)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def _run_pipe(**kwargs):
    """Call the Diffusers pipeline under the lock and return the first image.

    Also logs basic image statistics for debugging when possible.
    """
    # Best-effort: ensure scheduler has valid timesteps and step index
    try:
        steps = int(kwargs.get("num_inference_steps", 20))
    except Exception:
        steps = 20

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
            _log_call_begin_ext(steps, kwargs)
            with PIPE_LOCK:
                _prepare_scheduler_locked(int(steps))
                out = PIPE(**kwargs)
            _record_perf_ext(t0, kwargs)
            img_or_out = _image_or_passthrough_ext(out)
            _record_img0_stats_ext(out, img_or_out)
            return img_or_out
        except RuntimeError as e:
            if _should_retry_oom_ext(e, attempt, retries):
                attempt += 1
                continue
            raise


def _log_call_begin_ext(_steps: int, kwargs) -> None:
    pass  # Logging removed


def _record_perf_ext(_t0: float, kwargs) -> None:
    import time as _time

    dur_s = _time.perf_counter() - _t0
    try:
        LAST_CALL["dur_s"] = float(dur_s)
    except Exception:
        pass


def _image_or_passthrough_ext(_out):
    if hasattr(_out, "images") and getattr(_out, "images"):
        return _out.images[0]
    try:
        if _out is not None and not hasattr(_out, "images"):
            return _out
    except Exception:
        pass
    raise RuntimeError("Local FLUX pipeline returned no images")


def _record_img0_stats_ext(out, img_or_out) -> None:
    # If we have an image, record stats as before
    if hasattr(img_or_out, "__array__") or getattr(getattr(out, "images", []), "__len__", lambda: 0)():  # noqa: E501
        try:
            import numpy as _np  # type: ignore
            img0 = img_or_out if not hasattr(out, "images") else out.images[0]
            arr = _np.asarray(img0)
            LAST_CALL.update(
                {
                    "img0_mean": float(arr.mean()),
                    "img0_std": float(arr.std()),
                    "img0_min": float(arr.min()),
                    "img0_max": float(arr.max()),
                }
            )
            if _lv() >= 2:
                LOGGER.info(
                    "img0 stats mean=%.3f std=%.3f min=%s max=%s",
                    float(arr.mean()),
                    float(arr.std()),
                    arr.min(),
                    arr.max(),
                )
        except Exception:
            pass


def _should_retry_oom_ext(e: Exception, attempt: int, retries: int) -> bool:
    import time as _time
    try:
        msg = str(e).lower()
    except Exception:
        msg = ""
    if attempt < retries and "out of memory" in msg:
        try:
            import torch  # type: ignore

            torch.cuda.empty_cache()
        except Exception:
            pass
        _time.sleep(1.0)
        return True
    return False


def _post_load_toggles(pipe) -> None:
    """Enable lightweight perf toggles on a freshly loaded pipeline."""
    try:
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass


def _after_model_switch(mid: str) -> None:
    """Bookkeeping after switching models: clear caches, record logs."""
    # clear prompt cache on model switch
    try:
        PROMPT_CACHE.clear()
    except Exception:
        pass
    try:
        LAST_CALL.update({"event": "load_model", "model_id": mid})
        if _lv() >= 1:
            LOGGER.info("loaded model %s", mid)
    except Exception:
        pass


def _get_prompt_embeds(prompt: str, guidance: float):
    """Encode prompt once per model/prompt/guidance-mode and cache embeddings.

    For sd‑turbo we use the pipeline's encode_prompt to avoid re‑tokenizing
    and re‑encoding on every rerun. If encode_prompt isn't available, return
    (None, None) and fall back to string prompts in the caller.
    """
    if not guidance or guidance <= 1e-6:
        return (None, None)
    key = (CURRENT_MODEL_ID, prompt, bool(guidance and guidance > 1e-6))
    try:
        if key in PROMPT_CACHE:
            return PROMPT_CACHE[key]
        enc = getattr(PIPE, "encode_prompt", None)
        if enc is None:
            return (None, None)
        # num_images_per_prompt=1, classifier-free guidance as requested
        prompt_embeds, neg_embeds = enc(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=bool(key[2]),
            negative_prompt=None,
        )
        PROMPT_CACHE[key] = (prompt_embeds, neg_embeds)
        return PROMPT_CACHE[key]
    except Exception:
        return (None, None)


def generate_flux_image_latents(
    prompt: str,
    latents,
    width: int = 768,
    height: int = 768,
    steps: int | None = 20,
    guidance: float = 3.5,
):
    def _eff_guidance(mid: str, g: float) -> float:
        return _eff_g(mid, g)
    if steps is None:
        steps = 20
    # Remote image server path removed: always use local pipeline
    # Ensure pipeline is ready (env model id if not set)
    _ensure_pipe(None)

    # Convert and normalize latents succinctly
    latents = _to_cuda_fp16(latents)
    try:
        latents = _normalize_to_init_sigma(PIPE, latents, steps=int(steps))
    except TypeError:
        # test stubs may provide a 2-arg variant
        latents = _normalize_to_init_sigma(PIPE, latents)
    _record_latents_meta(latents, width, height, steps, guidance)
    guidance_eff = _eff_guidance(CURRENT_MODEL_ID or "", guidance)
    import sys as _sys
    _run = getattr(_sys.modules.get("flux_local"), "_run_pipe", _run_pipe)
    pe, ne = _get_prompt_embeds(prompt, guidance_eff)
    if pe is not None:
        return _run(
            prompt_embeds=pe,
            negative_prompt_embeds=ne,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_eff),
            width=int(width),
            height=int(height),
            latents=latents,
        )
    else:
        return _run(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_eff),
            width=int(width),
            height=int(height),
            latents=latents,
        )


def _record_latents_meta(latents, width: int, height: int, steps: int, guidance: float) -> None:
    """Record LAST_CALL + log a concise latents summary (pure helper)."""
    try:
        std, mean, shp = _latents_basic_stats(latents)
        guidance_eff = _eff_g(CURRENT_MODEL_ID or "", guidance)
        _update_last_call_latents(width, height, steps, guidance_eff, std, mean, shp)
        init_sigma = _get_init_sigma()
        if _lv() >= 2:
            LOGGER.info(
                "latents gen w=%s h=%s steps=%s g=%.3f std=%s mean=%s shape=%s init_sigma=%s",
                width,
                height,
                steps,
                guidance_eff,
                std,
                mean,
                shp,
                init_sigma,
            )
    except Exception:
        pass


def _latents_basic_stats(latents) -> tuple[float | None, float | None, tuple[int, ...] | None]:
    """Compute (std, mean, shape) for debug/logging; tolerate stubs."""
    try:
        std = float(latents.std().item()) if hasattr(latents, "std") else None
    except Exception:
        std = None
    try:
        mean = (
            float(getattr(latents, "mean")().item())
            if hasattr(latents, "mean")
            else None
        )
    except Exception:
        mean = None
    shp = None
    try:
        s = getattr(latents, "shape", None)
        if s is not None:
            shp = tuple(int(x) for x in s)
    except Exception:
        shp = None
    return std, mean, shp


def _update_last_call_latents(
    width: int,
    height: int,
    steps: int,
    guidance_eff: float,
    std,
    mean,
    shp,
) -> None:
    try:
        LAST_CALL.update(
            {
                "event": "latents_call",
                "model_id": CURRENT_MODEL_ID,
                "width": int(width),
                "height": int(height),
                "steps": int(steps),
                "guidance": float(guidance_eff),
                "latents_std": std,
                "latents_mean": mean,
                "latents_shape": shp,
            }
        )
    except Exception:
        pass


def _get_init_sigma():
    try:
        sched = getattr(PIPE, "scheduler", None)
        return getattr(sched, "init_noise_sigma", None)
    except Exception:
        return None


def set_model(model_id: str):
    """Load or switch the local FLUX/SD/SDXL model. CUDA only; no fallbacks.

    Protected by PIPE_LOCK so scheduler swaps don't race with in-flight calls.
    """
    with PIPE_LOCK:
        pipe = _ensure_pipe(model_id)
    # Use Euler A for SD‑Turbo unless already configured; improves stability
    try:
        if isinstance(model_id, str) and (
            "sd-turbo" in model_id or "sdxl-turbo" in model_id
        ):
            # SD‑Turbo models work best with LCM scheduler; EulerA can yield
            # degenerate outputs when injecting latents. Keep it simple.
            from diffusers import LCMScheduler  # type: ignore

            cfg = getattr(pipe, "scheduler", None).config
            pipe.scheduler = LCMScheduler.from_config(cfg)  # type: ignore[index]
            if _lv() >= 1:
                LOGGER.info("turbo model detected; using LCMScheduler for %s", model_id)
    except Exception:
        pass
def _get_model_id() -> str:
    """Return the effective default model id."""
    return _get_default_model_id()
