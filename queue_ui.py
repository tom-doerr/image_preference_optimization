from __future__ import annotations

import numpy as np


def _ensure_queue():
    import streamlit as st
    if 'queue' not in st.session_state:
        st.session_state.queue = []


def _queue_add_one():
    import streamlit as st
    import background as bg
    from latent_opt import z_to_latents
    from persistence import get_dataset_for_prompt_or_session
    from latent_opt import propose_next_pair

    lstate = st.session_state.lstate
    prompt = st.session_state.prompt
    steps = int(getattr(st.session_state, 'steps', 6))
    guidance_eff = float(getattr(st.session_state, 'guidance_eff', 0.0))

    from constants import DISTANCEHILL_GAMMA, COSINEHILL_BETA
    try:
        vmc = st.session_state.get('vm_choice', 'DistanceHill')
        pp = 'CosineHill' if vmc == 'CosineHill' else 'DistanceHill'
        Xd, yd = get_dataset_for_prompt_or_session(prompt, st.session_state)
        a = float(getattr(st.session_state, 'alpha', 0.5))
        if pp == 'DistanceHill':
            from latent_logic import propose_pair_distancehill
            za, _ = propose_pair_distancehill(lstate, prompt, Xd, yd, alpha=a, gamma=DISTANCEHILL_GAMMA, trust_r=None)
        elif pp == 'CosineHill':
            from latent_logic import propose_pair_cosinehill
            za, _ = propose_pair_cosinehill(lstate, prompt, Xd, yd, alpha=a, beta=COSINEHILL_BETA, trust_r=None)
        else:
            from app import _proposer_opts  # reuse app's opts builder
            za, _ = propose_next_pair(lstate, prompt, opts=_proposer_opts())
    except Exception:
        from latent_logic import z_from_prompt
        z_p = z_from_prompt(lstate, prompt)
        r = lstate.rng.standard_normal(lstate.d)
        r = r / (np.linalg.norm(r) + 1e-12)
        za = z_p + lstate.sigma * 0.8 * r
    lat = z_to_latents(lstate, za)
    fut = bg.schedule_decode_latents(prompt, lat, lstate.width, lstate.height, steps, guidance_eff)
    item = {'z': za, 'future': fut, 'label': None}
    q = st.session_state.get('queue') or []
    q.append(item)
    st.session_state.queue = q


def _queue_fill_up_to():
    import streamlit as st
    _ensure_queue()
    raw = getattr(st.session_state, 'queue_size', 6)
    size = int(raw if raw is not None else 6)
    while len(st.session_state.queue) < size:
        _queue_add_one()


def _queue_label(idx: int, label: int):
    import streamlit as st
    from batch_ui import _curation_add, _curation_train_and_next
    _ensure_queue()
    q = st.session_state.queue
    if 0 <= idx < len(q):
        z = q[idx]['z']
        _curation_add(int(label), z)
        try:
            _curation_train_and_next()  # retrain from disk; does not alter queue
        except Exception:
            pass
        q.pop(idx)
        st.session_state.queue = q


def _render_queue_ui():
    import streamlit as st
    st.subheader("Async queue")
    _ensure_queue()
    q = st.session_state.queue
    if not q:
        st.write("Queue empty…")
    else:
        i = 0
        it = q[0]
        img = it['future'].result() if it['future'].done() else None
        if img is not None:
            st.image(img, caption=f"Item {i}", use_container_width=True)
        else:
            st.write(f"Item {i}: loading…")
        if st.button(f"Accept {i}", use_container_width=True):
            _queue_label(i, 1)
            if callable(getattr(st, 'rerun', None)):
                try:
                    st.rerun()
                except Exception:
                    pass
        if st.button(f"Reject {i}", use_container_width=True):
            _queue_label(i, -1)
            if callable(getattr(st, 'rerun', None)):
                try:
                    st.rerun()
                except Exception:
                    pass
    _queue_fill_up_to()


def run_queue_mode() -> None:
    _ensure_queue()
    _queue_fill_up_to()
    _render_queue_ui()
