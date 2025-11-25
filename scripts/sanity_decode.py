import os
import sys
import numpy as np


def main():
    # Late imports to keep deps light if unused
    from ipo.infra import pipeline_local as fl

    model = os.getenv("SANITY_MODEL") or os.getenv("FLUX_LOCAL_MODEL") or "stabilityai/sd-turbo"
    prompt = os.getenv("SANITY_PROMPT", "latex, neon punk city, women with short hair, standing in the rain")
    width = int(os.getenv("SANITY_W", 512))
    height = int(os.getenv("SANITY_H", 512))
    steps = int(os.getenv("SANITY_STEPS", 6))

    print(f"[sanity] model={model} prompt={prompt!r} size={width}x{height} steps={steps}")
    fl.CURRENT_MODEL_ID = None  # ensure set_model will pick up env/default
    fl.set_model(model)
    img = fl.generate_flux_image(prompt, width=width, height=height, steps=steps, guidance=0.0)
    arr = np.asarray(img)
    std = float(arr.std())
    mean = float(arr.mean())
    print(f"[sanity] image stats mean={mean:.2f} std={std:.2f} min={arr.min()} max={arr.max()}")
    if std < 30.0:
        print("[sanity] FAIL: std < 30 (image likely flat/brown)")
        sys.exit(1)
    print("[sanity] OK")


if __name__ == "__main__":
    main()
