#!/usr/bin/env python3
import os
import time
from pathlib import Path

import numpy as np

from latent_opt import init_latent_state, z_to_latents
from flux_local import set_model, generate_flux_image_latents


def main():
    model = os.getenv("GEN_MODEL", "stabilityai/sd-turbo")
    width = int(os.getenv("GEN_W", "512"))
    height = int(os.getenv("GEN_H", "512"))
    steps = int(os.getenv("GEN_STEPS", "6"))
    guidance = float(os.getenv("GEN_GUIDE", "2.5"))
    prompt = os.getenv(
        "GEN_PROMPT", "latex, neon punk city, women with short hair, standing in the rain"
    )

    set_model(model)

    st = init_latent_state(width=width, height=height, d=32, seed=0)
    z = st.rng.standard_normal(st.d)
    za, zb = z, -z
    la = z_to_latents(st, za)
    lb = z_to_latents(st, zb)

    kw = dict(width=width, height=height, steps=steps, guidance=guidance)

    t0 = time.time()
    img_a = generate_flux_image_latents(prompt, latents=la, **kw)
    img_b = generate_flux_image_latents(prompt, latents=lb, **kw)
    dt = time.time() - t0

    outdir = Path("generated")
    outdir.mkdir(exist_ok=True)
    fa = outdir / "turbo_a.png"
    fb = outdir / "turbo_b.png"
    try:
        img_a.save(fa)
        img_b.save(fb)
    except Exception:
        pass

    A = np.asarray(img_a)
    B = np.asarray(img_b)
    mad = float(np.mean(np.abs(A.astype(np.float32) - B.astype(np.float32))))
    print(f"Model: {model}")
    print(f"Saved: {fa} ({fa.stat().st_size if fa.exists() else 'na'} bytes)")
    print(f"Saved: {fb} ({fb.stat().st_size if fb.exists() else 'na'} bytes)")
    print(
        f"A shape/mean/std/min/max: {A.shape} {A.mean():.2f} {A.std():.2f} {A.min()} {A.max()}"
    )
    print(
        f"B shape/mean/std/min/max: {B.shape} {B.mean():.2f} {B.std():.2f} {B.min()} {B.max()}"
    )
    print(f"MAD(A,B): {mad:.2f}")
    print(f"Total time: {dt:.2f}s for two images")


if __name__ == "__main__":
    main()
