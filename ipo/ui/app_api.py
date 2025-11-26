import numpy as _np
import streamlit as st

from ipo.infra.constants import DEFAULT_PROMPT, Keys
from ipo.ui import batch_ui as _batch_ui


def _export_state_bytes(state, prompt: str):
    from ipo.core.persistence import export_state_bytes as _rpc

    return _rpc(state, prompt)


def _init_pair_for_state(new_state) -> None:
    try:
        from latent_opt import propose_next_pair
        z1, z2 = propose_next_pair(new_state, st.session_state.prompt)
        st.session_state.lz_pair = (z1, z2)
        return
    except Exception:
        pass
    try:
        from latent_logic import propose_latent_pair_ridge
        st.session_state.lz_pair = propose_latent_pair_ridge(new_state)
        return
    except Exception:
        pass
    try:
        d = int(getattr(new_state, "d", 0))
        st.session_state.lz_pair = (_np.zeros(d, dtype=float), _np.zeros(d, dtype=float))
    except Exception:
        st.session_state.lz_pair = (None, None)


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
        from latent_logic import z_from_prompt as _zfp
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
    from .ui_sidebar import (
        render_model_decode_settings,
        render_modes_and_value_model,
        render_rows_and_last_action,
    )
    from .ui_sidebar import (
        render_sidebar_tail as render_sidebar_tail_module,
    )
    # number_input helper
    def safe_sidebar_num(_st, label, *, value, step=None, format=None):
        num = getattr(
            getattr(_st, "sidebar", _st),
            "number_input",
            getattr(_st, "number_input", None),
        )
        if callable(num):
            return num(label, value=value, step=step, format=format)
        return value
    vm_choice, selected_gen_mode, _batch_sz, _ = render_modes_and_value_model(st)
    render_rows_and_last_action(st, base_prompt, lstate)
    selected_model, width, height, steps, guidance, _apply_clicked = (
        render_model_decode_settings(st, lstate)
    )
    st.session_state[Keys.STEPS] = int(steps)
    st.session_state[Keys.GUIDANCE] = float(guidance)
    reg_lambda = safe_sidebar_num(st, "Ridge λ", value=1e300, step=0.1, format="%.6f") or 1e300
    st.session_state[Keys.REG_LAMBDA] = float(reg_lambda)
    eta_default = float(st.session_state.get(Keys.ITER_ETA) or 0.00001)
    iter_eta_num = (
        safe_sidebar_num(
            st,
            "Iterative step (eta)",
            value=eta_default,
            step=0.000000000001,
            format="%.12f",
        )
        or eta_default
    )
    st.session_state[Keys.ITER_ETA] = float(iter_eta_num)
    iter_eta = float(st.session_state.get(Keys.ITER_ETA) or eta_default)
    from ipo.infra.constants import DEFAULT_ITER_STEPS as _DEF_STEPS

    steps_default = int(st.session_state.get(Keys.ITER_STEPS) or _DEF_STEPS)
    iter_steps_num = (
        safe_sidebar_num(
            st, "Optimization steps (latent)", value=steps_default, step=1
        )
        or steps_default
    )
    st.session_state[Keys.ITER_STEPS] = int(iter_steps_num)
    iter_steps = int(st.session_state.get(Keys.ITER_STEPS) or steps_default)
    render_sidebar_tail_module(
        st,
        lstate,
        st.session_state.prompt,
        st.session_state.state_path,
        vm_choice,
        iter_steps,
        iter_eta,
        selected_model,
        apply_state_cb=lambda *a, **k: None,
        rerun_cb=lambda *a, **k: None,
    )
    st.session_state[Keys.GUIDANCE_EFF] = 0.0
    return (
        vm_choice,
        selected_gen_mode,
        selected_model,
        int(width),
        int(height),
        int(steps),
        float(guidance),
        float(reg_lambda),
        int(iter_steps),
        float(iter_eta),
        False,
    )


def generate_pair(base_prompt: str) -> None:
    from ipo.infra.pipeline_local import generate_flux_image_latents as _gen
    from latent_opt import z_to_latents as _z2l
    try:
        lstate = st.session_state.lstate
        if st.session_state.get("lz_pair") is None:
            from latent_logic import z_from_prompt

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


def render_sidebar_tail(
    st_mod,
    lstate,
    prompt: str,
    state_path: str,
    vm_choice: str,
    iter_steps: int,
    iter_eta: float | None,
    selected_model: str,
    apply_state_cb,
    rerun_cb,
) -> None:
    """Proxy to ui_sidebar.render_sidebar_tail to avoid app.py direct imports."""
    try:
        from .ui_sidebar import render_sidebar_tail as _rst

        _rst(
            st_mod,
            lstate,
            prompt,
            state_path,
            vm_choice,
            iter_steps,
            iter_eta,
            selected_model,
            apply_state_cb,
            rerun_cb,
        )
    except Exception:
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
            from latent_logic import z_from_prompt as _zfp

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
            from latent_logic import z_from_prompt as _zfp

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
