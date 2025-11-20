from __future__ import annotations

import numpy as np
from typing import Any

import streamlit as st
from PIL import Image

from constants import Keys, Config, DEFAULT_MODEL


def _ensure_model() -> None:
    try:
        from flux_local import CURRENT_MODEL_ID, set_model  # type: ignore
        if CURRENT_MODEL_ID is None:
            set_model(DEFAULT_MODEL)
    except Exception:
        pass


def _img_to_array(img: Image.Image, w: int, h: int) -> np.ndarray:
    arr = np.asarray(img.convert("RGB").resize((w, h))).astype(np.float32) / 255.0
    return arr


def _decode_from_mu(mu: np.ndarray, w: int, h: int, steps: int, guidance: float) -> Any:
    from flux_local import generate_flux_image_latents  # local import for tests
    h8, w8 = h // 8, w // 8
    lat = mu.reshape(1, 4, h8, w8).astype(np.float32)
    return generate_flux_image_latents("image match", latents=lat, width=w, height=h, steps=steps, guidance=guidance)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _step_opt(mu: np.ndarray, target: np.ndarray, w: int, h: int, steps: int, guidance: float, alpha: float = 0.15, k: int = 2, rng: np.random.Generator | None = None) -> tuple[np.ndarray, Any, float]:
    rng = rng or np.random.default_rng()
    best_mu = mu
    best_img = _decode_from_mu(mu, w, h, steps, guidance)
    best_arr = _img_to_array(best_img, w, h)
    best_mse = _mse(best_arr, target)
    d = mu.size
    for _ in range(int(k)):
        dlt = rng.standard_normal(d).astype(np.float32)
        n = float(np.linalg.norm(dlt)) or 1.0
        dlt = (alpha / n) * dlt
        for sgn in (1.0, -1.0):
            cand_mu = mu + float(sgn) * dlt
            cand_img = _decode_from_mu(cand_mu, w, h, steps, guidance)
            cand_arr = _img_to_array(cand_img, w, h)
            m = _mse(cand_arr, target)
            if m < best_mse:
                best_mse = m
                best_mu = cand_mu
                best_img = cand_img
    return best_mu, best_img, best_mse


def _init_state(w: int, h: int) -> None:
    h8, w8 = h // 8, w // 8
    st.session_state.setdefault(Keys.IMATCH_MU, np.zeros((4 * h8 * w8,), dtype=np.float32))
    st.session_state.setdefault(Keys.IMATCH_LAST_IMG, None)
    st.session_state.setdefault(Keys.IMATCH_LAST_MSE, None)


def _main() -> None:
    st.subheader("Image match (latents)")
    _ensure_model()

    # Basic controls
    w = int(st.sidebar.number_input("Width", min_value=128, max_value=1024, value=512, step=64))
    h = int(st.sidebar.number_input("Height", min_value=128, max_value=1024, value=512, step=64))
    steps = int(st.sidebar.slider("Steps", min_value=1, max_value=30, value=Config.DEFAULT_STEPS))
    guidance = float(st.sidebar.number_input("Guidance", min_value=0.0, max_value=10.0, value=0.0, step=0.5))
    alpha = float(st.sidebar.slider("Step size (alpha)", min_value=0.01, max_value=0.5, value=0.15, step=0.01))
    k = int(st.sidebar.slider("Candidates per step", min_value=1, max_value=6, value=2))

    _init_state(w, h)
    upl = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)
    if not upl:
        st.write("Upload an image to start.")
        return
    try:
        img = Image.open(upl)
    except Exception:
        st.write("Invalid image.")
        return
    tgt_arr = _img_to_array(img, w, h)
    st.session_state[Keys.IMATCH_TARGET] = tgt_arr

    # Current attempt
    mu = st.session_state[Keys.IMATCH_MU]
    if st.session_state.get(Keys.IMATCH_LAST_IMG) is None:
        st.session_state[Keys.IMATCH_LAST_IMG] = _decode_from_mu(mu, w, h, steps, guidance)
        st.session_state[Keys.IMATCH_LAST_MSE] = _mse(_img_to_array(st.session_state[Keys.IMATCH_LAST_IMG], w, h), tgt_arr)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original", width="stretch")
    with col2:
        st.image(st.session_state[Keys.IMATCH_LAST_IMG], caption=f"Last attempt (MSE={st.session_state[Keys.IMATCH_LAST_MSE]:.4f})", width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Step"):
            mu, im, mse = _step_opt(mu, tgt_arr, w, h, steps, guidance, alpha=alpha, k=k)
            st.session_state[Keys.IMATCH_MU] = mu
            st.session_state[Keys.IMATCH_LAST_IMG] = im
            st.session_state[Keys.IMATCH_LAST_MSE] = mse
            try:
                print(f"[imatch] step mse={mse:.6f}")
            except Exception:
                pass
    with c2:
        if st.button("Auto ×5"):
            for _ in range(5):
                mu, im, mse = _step_opt(mu, tgt_arr, w, h, steps, guidance, alpha=alpha, k=k)
                st.session_state[Keys.IMATCH_MU] = mu
                st.session_state[Keys.IMATCH_LAST_IMG] = im
                st.session_state[Keys.IMATCH_LAST_MSE] = mse
            try:
                print(f"[imatch] auto5 mse={mse:.6f}")
            except Exception:
                pass
    with c3:
        if st.button("Reset"):
            st.session_state[Keys.IMATCH_MU] = np.zeros_like(st.session_state[Keys.IMATCH_MU])
            st.session_state[Keys.IMATCH_LAST_IMG] = _decode_from_mu(st.session_state[Keys.IMATCH_MU], w, h, steps, guidance)
            st.session_state[Keys.IMATCH_LAST_MSE] = _mse(_img_to_array(st.session_state[Keys.IMATCH_LAST_IMG], w, h), tgt_arr)


_main()

