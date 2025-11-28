import numpy as _np
import streamlit as st

from ipo.infra.constants import DEFAULT_ITER_ETA, DEFAULT_ITER_STEPS, DEFAULT_PROMPT, Keys
from ipo.ui import batch_ui as _batch_ui


def _export_state_bytes(state, prompt: str):
    from ipo.core.persistence import export_state_bytes as _rpc

    return _rpc(state, prompt)


def _init_pair_for_state(new_state) -> None:
    try:
        from ipo.core.latent_state import propose_pair_prompt_anchor
        z1, z2 = propose_pair_prompt_anchor(new_state, st.session_state.prompt)
        st.session_state.lz_pair = (z1, z2)
    except Exception:
        d = int(getattr(new_state, "d", 0))
        st.session_state.lz_pair = (_np.zeros(d), _np.zeros(d))


def _reset_derived_state(new_state) -> None:
    st.session_state[Keys.IMAGES] = (None, None)
    st.session_state[Keys.MU_IMAGE] = None
    if getattr(new_state, "mu", None) is None:
        new_state.mu = _np.zeros(int(getattr(new_state, "d", 0)), dtype=float)
    _mh = getattr(new_state, "mu_hist", None) or []
    st.session_state.mu_history = [m.copy() for m in _mh] or [new_state.mu.copy()]
    st.session_state.mu_best_idx = 0
    st.session_state.prompt_image = None
    for k in ("next_prefetch", "_bg_exec"):
        st.session_state.pop(k, None)


def _randomize_mu_if_zero(st_local, new_state) -> None:
    try:
        from ipo.core.latent_state import z_from_prompt as _zfp
        if _np.allclose(new_state.mu, 0.0):
            pr = (
                st_local.session_state.get(Keys.PROMPT)
                or st_local.session_state.get("prompt")
                or DEFAULT_PROMPT
            )
            z_p = _zfp(new_state, pr)
            r = new_state.rng.standard_normal(new_state.d).astype(float)
            nr = float(_np.linalg.norm(r))
            if nr > 0.0:
                r = r / nr
            new_state.mu = z_p + float(new_state.sigma) * r
    except Exception:
        pass


def _apply_state(*args) -> None:
    # Flexible: _apply_state(new_state) or _apply_state(st, new_state)
    if len(args) == 1:
        st_local, new_state = st, args[0]
    elif len(args) == 2:
        st_local, new_state = args  # type: ignore[misc]
    else:
        raise TypeError("_apply_state() expects 1 or 2 arguments")

    st_local.session_state.lstate = new_state
    try:
        use_rand = bool(getattr(st_local.session_state, Keys.USE_RANDOM_ANCHOR, False))
        setattr(new_state, "use_random_anchor", use_rand)
        setattr(new_state, "random_anchor_z", None)
    except Exception:
        pass
    _init_pair_for_state(new_state)
    _reset_derived_state(new_state)
    # Random μ init around the prompt anchor when μ is all zeros.
    _randomize_mu_if_zero(st_local, new_state)


def build_controls(st, lstate, base_prompt):
    vm_choice = st.session_state.get(Keys.VM_CHOICE) or "XGBoost"
    width, height = int(getattr(lstate, "width", 512)), int(getattr(lstate, "height", 512))
    steps = int(st.session_state.get(Keys.STEPS) or 6)
    guidance = float(st.session_state.get(Keys.GUIDANCE) or 0.0)
    reg_lambda = float(st.session_state.get(Keys.REG_LAMBDA) or 1000)
    iter_eta = float(st.session_state.get(Keys.ITER_ETA) or DEFAULT_ITER_ETA)
    iter_steps = int(st.session_state.get(Keys.ITER_STEPS) or DEFAULT_ITER_STEPS)
    st.session_state[Keys.GUIDANCE_EFF] = guidance
    return (vm_choice, "batch", None, width, height, steps, guidance,
            reg_lambda, iter_steps, iter_eta, False)


def generate_pair(base_prompt: str) -> None:
    from ipo.core.latent_state import z_to_latents as _z2l
    from ipo.infra.pipeline_local import generate_flux_image_latents as _gen
    try:
        lstate = st.session_state.lstate
        if st.session_state.get("lz_pair") is None:
            from ipo.core.latent_state import z_from_prompt

            z_p = z_from_prompt(lstate, base_prompt)
            r = lstate.rng.standard_normal(lstate.d)
            r = r / (float(_np.linalg.norm(r)) + 1e-12)
            delta = float(lstate.sigma) * 0.5 * r
            st.session_state.lz_pair = (z_p + delta, z_p - delta)
        z_a, z_b = st.session_state.lz_pair
        la = _z2l(lstate, z_a)
        lb = _z2l(lstate, z_b)
        img_a = _gen(
            base_prompt,
            latents=la,
            width=lstate.width,
            height=lstate.height,
            steps=int(st.session_state.get("steps", 6) or 6),
            guidance=float(st.session_state.get("guidance_eff", 0.0) or 0.0),
        )
        img_b = _gen(
            base_prompt,
            latents=lb,
            width=lstate.width,
            height=lstate.height,
            steps=int(st.session_state.get("steps", 6) or 6),
            guidance=float(st.session_state.get("guidance_eff", 0.0) or 0.0),
        )
        st.session_state[Keys.IMAGES] = (img_a, img_b)
    except Exception:
        pass


def render_sidebar_tail(*_args, **_kwargs) -> None:
    """Sidebar removed; no-op stub for backwards compatibility."""
    pass


def _render_batch_ui() -> None:
    return _batch_ui._render_batch_ui()


def _curation_init_batch() -> None:
    try:
        _batch_ui._curation_init_batch()
    except Exception:
        pass
    try:
        if not getattr(st.session_state, "cur_batch", None):
            from ipo.core.latent_state import z_from_prompt as _zfp

            z_p = _zfp(st.session_state.lstate, st.session_state.prompt)
            n = int(getattr(st.session_state, "batch_size", 4))
            rng = _np.random.default_rng(0)
            zs = [z_p + 0.01 * rng.standard_normal(z_p.shape) for _ in range(n)]
            st.session_state.cur_batch = zs
            st.session_state.cur_labels = [None] * n
    except Exception:
        pass


def _curation_new_batch() -> None:
    try:
        _batch_ui._curation_new_batch()
    except Exception:
        pass
    try:
        if not getattr(st.session_state, "cur_batch", None):
            _curation_init_batch()
    except Exception:
        pass


def _curation_replace_at(idx: int) -> None:
    try:
        _batch_ui._curation_replace_at(idx)
    except Exception:
        pass
    try:
        zs = getattr(st.session_state, "cur_batch", None)
        if isinstance(zs, list) and len(zs) > 0:
            from ipo.core.latent_state import z_from_prompt as _zfp

            z_p = _zfp(st.session_state.lstate, st.session_state.prompt)
            rng = _np.random.default_rng(idx + 1)
            zs[idx % len(zs)] = z_p + 0.01 * rng.standard_normal(z_p.shape)
            st.session_state.cur_batch = zs
    except Exception:
        pass


def _curation_add(label: int, z, img=None) -> None:
    try:
        return _batch_ui._curation_add(label, z, img)
    except Exception:
        return None


def _curation_train_and_next() -> None:
    try:
        return _batch_ui._curation_train_and_next()
    except Exception:
        return None


def run_app(_st, _vm_choice: str, _selected_gen_mode: str | None) -> None:
    _batch_ui.run_batch_mode()
