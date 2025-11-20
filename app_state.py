from __future__ import annotations

from typing import Any


def _init_pair_for_state(st: Any, new_state: Any) -> None:
    try:
        vmc = st.session_state.get("vm_choice", "DistanceHill")
        pp = "CosineHill" if vmc == "CosineHill" else "DistanceHill"
        if pp == "DistanceHill":
            from persistence import get_dataset_for_prompt_or_session
            from latent_logic import propose_pair_distancehill

            Xd, yd = get_dataset_for_prompt_or_session(st.session_state.prompt, st.session_state)
            z1, z2 = propose_pair_distancehill(new_state, st.session_state.prompt, Xd, yd, alpha=0.5, gamma=0.5, trust_r=None)
        elif pp == "CosineHill":
            from persistence import get_dataset_for_prompt_or_session
            from latent_logic import propose_pair_cosinehill

            Xd, yd = get_dataset_for_prompt_or_session(st.session_state.prompt, st.session_state)
            z1, z2 = propose_pair_cosinehill(new_state, st.session_state.prompt, Xd, yd, alpha=0.5, beta=5.0, trust_r=None)
        else:
            from latent_opt import propose_next_pair

            z1, z2 = propose_next_pair(new_state, st.session_state.prompt)
        st.session_state.lz_pair = (z1, z2)
    except Exception:
        from latent_logic import propose_latent_pair_ridge

        st.session_state.lz_pair = propose_latent_pair_ridge(new_state)


def _reset_derived_state(st: Any, new_state: Any) -> None:
    from constants import Keys
    import numpy as _np
    st.session_state[Keys.IMAGES] = (None, None)
    st.session_state[Keys.MU_IMAGE] = None
    if getattr(new_state, "mu", None) is None:
        setattr(new_state, "mu", _np.zeros(int(getattr(new_state, "d", 0)), dtype=float))
    if getattr(new_state, "mu_hist", None) is not None and new_state.mu_hist.size > 0:
        st.session_state.mu_history = [m.copy() for m in new_state.mu_hist]
    else:
        st.session_state.mu_history = [new_state.mu.copy()]
    st.session_state.mu_best_idx = 0
    st.session_state.prompt_image = None
    for k in ("next_prefetch", "_bg_exec"):
        st.session_state.pop(k, None)
    try:
        from background import reset_executor

        reset_executor()
    except Exception:
        pass


def _apply_state(st: Any, new_state: Any) -> None:
    from constants import Keys
    st.session_state.lstate = new_state
    try:
        use_rand = bool(getattr(st.session_state, Keys.USE_RANDOM_ANCHOR, False))
        setattr(new_state, "use_random_anchor", use_rand)
        setattr(new_state, "random_anchor_z", None)
    except Exception:
        pass
    _init_pair_for_state(st, new_state)
    _reset_derived_state(st, new_state)
