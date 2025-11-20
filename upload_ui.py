from __future__ import annotations

import hashlib
import os
from typing import Any

import streamlit as st
from PIL import Image

from constants import Keys
from flux_local import generate_flux_image_latents
from img_latents import image_to_z
from latent_opt import z_from_prompt


def run_upload_mode(lstate: Any, prompt: str) -> None:
    st.subheader("Upload latents")

    from value_scorer import get_value_scorer_with_status

    scorer, scorer_status = get_value_scorer_with_status(
        st.session_state.get("vm_choice"), lstate, prompt, st.session_state
    )

    uploads = getattr(st.sidebar, "file_uploader", lambda *a, **k: [])(
        "Upload images to use as latents",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "webp"],
    )
    steps = int(getattr(st.session_state, Keys.STEPS, 6))
    guidance_eff = float(getattr(st.session_state, Keys.GUIDANCE, 0.0))
    z_p = z_from_prompt(lstate, prompt)
    nonce = int(st.session_state.get(Keys.CUR_BATCH_NONCE, 0))

    if not uploads:
        st.write("Upload at least one image to score it as Good/Bad.")
        return

    def _save_upload_image(img_raw, n: int, idx: int) -> None:
        try:
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
            up_dir = os.path.join("data", h, "uploads")
            os.makedirs(up_dir, exist_ok=True)
            fname = f"upload_{n}_{idx}.png"
            img_raw.save(os.path.join(up_dir, fname))
        except Exception:
            pass

    for idx, upl in enumerate(uploads):
        try:
            img_raw = Image.open(upl)
        except Exception:
            continue
        _save_upload_image(img_raw, nonce, idx)
        z_upl = image_to_z(img_raw, lstate)
        alpha_interp = st.slider(
            f"Interpolate toward prompt (α) {idx}", value=1.0, step=0.05, key=f"upl_interp_{nonce}_{idx}"
        )
        z = (1.0 - float(alpha_interp)) * z_p + float(alpha_interp) * z_upl
        lat = z.reshape(1, 4, lstate.height // 8, lstate.width // 8)
        img_dec = generate_flux_image_latents(
            prompt, latents=lat, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff
        )
        st.image(img_dec, caption=f"Upload {idx}", width="stretch")
        try:
            if scorer is not None and scorer_status == "ok":
                score_val = float(scorer(z - z_p))
                st.caption(f"Score: {score_val:.3f}")
            else:
                st.caption("Score: n/a")
        except Exception:
            pass
        # Weight + Good/Bad buttons
        w = st.slider(f"Weight upload {idx}", value=1.0, step=0.1, key=f"upl_w_{nonce}_{idx}")
        try:
            import batch_ui as _bu  # local import for tests

            if st.button(f"Good (+1) upload {idx}", key=f"upl_good_{nonce}_{idx}"):
                _bu._curation_add(float(w), z, img=None)
                _bu._curation_train_and_next()
            if st.button(f"Bad (-1) upload {idx}", key=f"upl_bad_{nonce}_{idx}"):
                _bu._curation_add(-float(w), z, img=None)
                _bu._curation_train_and_next()
        except Exception:
            pass
