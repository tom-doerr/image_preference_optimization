from __future__ import annotations

import math
import os
from typing import Any, Optional


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


def _scheduler_init_sigma(pipe: Any, steps: Optional[int]) -> float:
    try:
        sched = getattr(pipe, "scheduler", None)
        if sched is None:
            return 1.0
        _ensure_timesteps(sched, steps)
        s = getattr(sched, "init_noise_sigma", None)
        if s is None:
            s = _sigma_from_sigmas_attr(sched)
        return float(s) if s is not None else 1.0
    except Exception:
        return 1.0


def _ensure_timesteps(sched: Any, steps: Optional[int]) -> None:
    if isinstance(steps, int) and steps > 0 and hasattr(sched, "set_timesteps"):
        try:
            sched.set_timesteps(int(steps))
        except Exception:
            pass


def _sigma_from_sigmas_attr(sched: Any):
    sigmas = getattr(sched, "sigmas", None)
    if sigmas is not None and hasattr(sigmas, "max"):
        try:
            return float(sigmas.max().item())
        except Exception:
            return None
    return None


def _latents_std(latents) -> float:
    try:
        if hasattr(latents, "numel") and hasattr(latents, "std"):
            return float(latents.std().item()) if latents.numel() else 1.0
    except Exception:
        pass
    try:
        import numpy as _np  # type: ignore

        arr = getattr(latents, "arr", None)
        val = float(_np.asarray(arr).std()) if arr is not None else 1.0
    except Exception:
        val = 1.0
    if not math.isfinite(val) or val == 0.0:
        val = 1.0
    return val


def normalize_to_init_sigma(pipe: Any, latents, steps: Optional[int] = None):
    """Scale latents so std ≈ scheduler.init_noise_sigma for current steps.

    Works with real torch tensors and simple numpy‑backed stubs.
    """
    init_sigma = _scheduler_init_sigma(pipe, steps)
    std = _latents_std(latents)
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
