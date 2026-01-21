import logging
import numpy as _np
import streamlit as st

from ipo.infra.constants import DEFAULT_ITER_ETA, DEFAULT_ITER_STEPS, DEFAULT_PROMPT, Keys
from ipo.ui import batch_ui as _batch_ui

log = logging.getLogger(__name__)


def _init_pair_for_state(new_state) -> None:
    try:
        from ipo.core.latent_state import propose_pair_prompt_anchor
        z1, z2 = propose_pair_prompt_anchor(new_state, st.session_state.prompt)
        st.session_state.lz_pair = (z1, z2)
    except Exception as e:
        log.warning("_init_pair_for_state failed: %s", e)
        d = int(getattr(new_state, "d", 0))
        st.session_state.lz_pair = (_np.zeros(d), _np.zeros(d))


def _reset_derived_state(new_state) -> None:
    st.session_state[Keys.IMAGES] = (None, None)
    st.session_state["mu_image"] = None
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
    except Exception as e:
        log.warning("_randomize_mu_if_zero failed: %s", e)


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
    except Exception as e:
        log.debug("_apply_state anchor setup: %s", e)
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


def run_app(_st, _vm_choice: str, _selected_gen_mode: str | None) -> None:
    _batch_ui.run_batch_mode()
