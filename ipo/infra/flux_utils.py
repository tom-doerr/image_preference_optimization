from __future__ import annotations

from typing import Optional, Any
import os
import math


def get_default_model_id() -> str:
    """Return default model id for local pipeline.

    Respects FLUX_LOCAL_MODEL, otherwise falls back to sd‑turbo.
    """
    mid = os.getenv("FLUX_LOCAL_MODEL")
    return mid or "stabilityai/sd-turbo"


def _env_log_verbosity() -> int:
    try:
        return int(os.getenv("LOG_VERBOSITY", "0"))
    except Exception:
        return 0


def log_verbosity() -> int:
    try:
        # Prefer shared util helper when present
        from ipo.infra.util import get_log_verbosity as _get

        return int(_get(None))
    except Exception:  # pragma: no cover
        return _env_log_verbosity()


def p(msg: str, lvl: int = 1) -> None:
    if log_verbosity() >= int(lvl):
        try:
            print(msg)
        except Exception:  # pragma: no cover
            pass


def to_cuda_fp16(latents):
    """Return a torch fp16 CUDA tensor for the given latents (np/torch)."""
    try:
        import torch  # type: ignore

        TensorType = getattr(torch, "Tensor", None)
        if TensorType is not None and isinstance(latents, TensorType):
            return latents.to(device="cuda", dtype=torch.float16)
        to_tensor = getattr(torch, "tensor", None)
        if callable(to_tensor):
            return to_tensor(latents, dtype=torch.float16, device="cuda")
    except Exception:
        pass
    # As a last resort (tests without full torch), return input unchanged
    return latents


def normalize_to_init_sigma(pipe: Any, latents, steps: Optional[int] = None):
    """Scale latents so std ≈ scheduler.init_noise_sigma for current steps.

    Works with real torch tensors and simple numpy‑backed stubs.
    """
    init_sigma = 1.0
    try:
        sched = getattr(pipe, "scheduler", None)
        if sched is not None:
            if isinstance(steps, int) and steps > 0 and hasattr(sched, "set_timesteps"):
                try:
                    sched.set_timesteps(int(steps))
                except Exception:
                    pass
            s = getattr(sched, "init_noise_sigma", None)
            if s is None:
                sigmas = getattr(sched, "sigmas", None)
                if sigmas is not None and hasattr(sigmas, "max"):
                    s = float(sigmas.max().item())
            init_sigma = float(s) if s is not None else 1.0
    except Exception:
        pass
    try:
        std = float(latents.std().item()) if latents.numel() else 1.0
    except Exception:
        try:
            import numpy as _np  # type: ignore

            arr = getattr(latents, "arr", None)
            std = float(_np.asarray(arr).std()) if arr is not None else 1.0
        except Exception:
            std = 1.0
    if not math.isfinite(std) or std == 0.0:
        std = 1.0
    try:
        return (latents / std) * init_sigma
    except Exception:
        arr = getattr(latents, "arr", None)
        if arr is not None:
            latents.arr = (arr / std) * init_sigma
            return latents
        return latents


def eff_guidance(model_id: str, guidance: float) -> float:
    if isinstance(model_id, str) and ("sd-turbo" in model_id or "sdxl-turbo" in model_id):
        return 0.0
    return guidance
