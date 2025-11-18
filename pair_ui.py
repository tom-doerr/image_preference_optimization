from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def _lstate_and_prompt():
    import streamlit as st
    try:
        prompt = st.session_state.get('prompt')
    except Exception:
        prompt = None
    if not prompt:
        from constants import DEFAULT_PROMPT
        prompt = DEFAULT_PROMPT
    return st.session_state.lstate, prompt


def _decode_one(side: str, latents: np.ndarray) -> Any:
    import streamlit as st
    from flux_local import generate_flux_image_latents
    lstate, prompt = _lstate_and_prompt()
    img = generate_flux_image_latents(
        prompt,
        latents=latents,
        width=lstate.width,
        height=lstate.height,
        steps=int(st.session_state.get('steps', 6)),
        guidance=float(st.session_state.get('guidance_eff', 0.0)),
    )
    try:
        # Optional stats collection; ignored in stubs that don't expose get_last_call
        from flux_local import get_last_call  # type: ignore
        st.session_state.img_stats = st.session_state.get('img_stats') or {}
        st.session_state.img_stats[side] = get_last_call().copy()
    except Exception:
        pass
    return img


def _prefetch_next_for_generate() -> None:
    import streamlit as st
    import background as bg
    from latent_opt import z_to_latents, propose_next_pair
    from latent_logic import propose_pair_distancehill, propose_pair_cosinehill, propose_latent_pair_ridge
    from persistence import get_dataset_for_prompt_or_session

    lstate, prompt = _lstate_and_prompt()
    vmc = st.session_state.get('vm_choice')
    try:
        if vmc == 'DistanceHill' or vmc == 'CosineHill':
            Xd, yd = get_dataset_for_prompt_or_session(prompt, st.session_state)
            a = float(st.session_state.get('alpha', 0.5))
            if vmc == 'DistanceHill':
                za_n, zb_n = propose_pair_distancehill(lstate, prompt, Xd, yd, alpha=a, gamma=0.5, trust_r=None)
            else:
                za_n, zb_n = propose_pair_cosinehill(lstate, prompt, Xd, yd, alpha=a, beta=5.0, trust_r=None)
        else:
            # Fallback to generic proposer
            from app import _proposer_opts  # reuse current opts
            za_n, zb_n = propose_next_pair(lstate, prompt, opts=_proposer_opts())
    except Exception:
        za_n, zb_n = propose_latent_pair_ridge(lstate)
    la_n = z_to_latents(lstate, za_n)
    lb_n = z_to_latents(lstate, zb_n)
    f = bg.schedule_decode_pair(
        prompt,
        la_n,
        lb_n,
        lstate.width,
        lstate.height,
        int(st.session_state.get('steps', 6)),
        float(st.session_state.get('guidance_eff', 0.0)),
    )
    st.session_state.next_prefetch = {'za': za_n, 'zb': zb_n, 'f': f}


def generate_pair() -> None:
    import streamlit as st
    from latent_opt import z_to_latents

    lstate, prompt = _lstate_and_prompt()
    if st.session_state.get('lz_pair') is None:
        # Minimal init if missing: symmetric pair around prompt anchor
        from latent_logic import z_from_prompt
        z_p = z_from_prompt(lstate, prompt)
        r = lstate.rng.standard_normal(lstate.d)
        r = r / (np.linalg.norm(r) + 1e-12)
        delta = lstate.sigma * 0.5 * r
        st.session_state.lz_pair = (z_p + delta, z_p - delta)
    z_a, z_b = st.session_state.lz_pair
    lat_a = z_to_latents(lstate, z_a)
    lat_b = z_to_latents(lstate, z_b)
    img_a = _decode_one('left', lat_a)
    img_b = _decode_one('right', lat_b)
    st.session_state.images = (img_a, img_b)
    _prefetch_next_for_generate()


def _pair_scores() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    import streamlit as st
    from latent_logic import z_from_prompt
    from value_scorer import get_value_scorer

    lstate, prompt = _lstate_and_prompt()
    try:
        z_a, z_b = st.session_state.lz_pair
        z_p = z_from_prompt(lstate, prompt)
        d_left = float(np.linalg.norm(z_a - z_p))
        d_right = float(np.linalg.norm(z_b - z_p))
        scorer = get_value_scorer(st.session_state.get('vm_choice'), lstate, prompt, st.session_state)
        v_left = float(scorer(z_a - z_p))
        v_right = float(scorer(z_b - z_p))
        return d_left, d_right, v_left, v_right
    except Exception:
        return None, None, None, None
