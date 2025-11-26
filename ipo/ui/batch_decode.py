from __future__ import annotations

from typing import Any


def decode_one(i: int, lstate: Any, prompt: str, z_i, steps: int, guidance_eff: float):
    """Decode a single latent vector to an image via flux_local.

    Extracted from batch_ui to reduce file complexity; behavior unchanged.
    """
    import time as _time
    try:
        from latent_logic import z_to_latents
    except Exception:
        from latent_opt import z_to_latents  # tests may stub here
    try:
        from ipo.infra.pipeline_local import generate_flux_image_latents
    except Exception:
        # Allow tests that inject a stub module to satisfy this import
        from sys import modules as _modules
        _fl = _modules.get("flux_local")
        generate_flux_image_latents = getattr(  # type: ignore
            _fl, "generate_flux_image_latents"
        )

    t0 = _time.perf_counter()
    try:
        la = z_to_latents(lstate, z_i)
    except Exception:
        la = z_to_latents(z_i, lstate)
    img_i = generate_flux_image_latents(
        prompt,
        latents=la,
        width=lstate.width,
        height=lstate.height,
        steps=steps,
        guidance=guidance_eff,
    )
    # Optional: callers may log perf; keep function focused
    try:
        dt_ms = (_time.perf_counter() - t0) * 1000.0
        from .batch_ui import _log  # local import to avoid cycles on import time
        _log(
            (
                f"[batch] decoded item={i} in {dt_ms:.1f} ms "
                f"(steps={steps}, w={lstate.width}, h={lstate.height})"
            )
        )
    except Exception:
        pass
    return img_i
