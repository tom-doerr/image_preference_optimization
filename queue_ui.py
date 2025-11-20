from __future__ import annotations

import numpy as np
from constants import Keys

__all__ = [
    '_ensure_queue',
    '_queue_add_one',
    '_queue_fill_up_to',
    '_queue_label',
    '_render_queue_ui',
    'run_queue_mode',
]


def _ensure_queue() -> None:
    import streamlit as st
    if Keys.QUEUE not in st.session_state:
        st.session_state[Keys.QUEUE] = []


def _queue_add_one() -> None:
    import streamlit as st
    import background as bg
    from latent_opt import z_to_latents
    from persistence import get_dataset_for_prompt_or_session
    from latent_opt import propose_next_pair
    from latent_state import init_latent_state

    lstate = getattr(st.session_state, 'lstate', None)
    if lstate is None:
        lstate = init_latent_state()
        st.session_state.lstate = lstate
    prompt = getattr(st.session_state, 'prompt', None)
    if not prompt:
        from constants import DEFAULT_PROMPT
        prompt = DEFAULT_PROMPT
    raw_steps = getattr(st.session_state, Keys.STEPS, 6)
    steps = int(raw_steps if raw_steps is not None else 6)
    raw_guid = getattr(st.session_state, Keys.GUIDANCE_EFF, 0.0)
    guidance_eff = float(raw_guid if raw_guid is not None else 0.0)

    from constants import DISTANCEHILL_GAMMA, COSINEHILL_BETA
    try:
        vmc = st.session_state.get(Keys.VM_CHOICE, 'DistanceHill')
        pp = 'CosineHill' if vmc == 'CosineHill' else 'DistanceHill'
        Xd, yd = get_dataset_for_prompt_or_session(prompt, st.session_state)
        # persistence.get_dataset_for_prompt_or_session already guards dim mismatches
        raw_alpha = getattr(st.session_state, 'alpha', 0.5)
        a = float(raw_alpha if raw_alpha is not None else 0.5)
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
        # Reuse batch helper to keep sampling logic in one place
        try:
            from batch_ui import _sample_around_prompt as _sap  # local import to avoid cycles
            za = _sap(scale=0.8)
        except Exception:
            from latent_logic import z_from_prompt
            z_p = z_from_prompt(lstate, prompt)
            r = lstate.rng.standard_normal(lstate.d)
            r = r / (np.linalg.norm(r) + 1e-12)
            za = z_p + lstate.sigma * 0.8 * r
    lat = z_to_latents(lstate, za)
    fut = bg.schedule_decode_latents(prompt, lat, lstate.width, lstate.height, steps, guidance_eff)
    item = {'z': za, 'future': fut, 'label': None}
    q = st.session_state.get(Keys.QUEUE) or []
    q.append(item)
    st.session_state[Keys.QUEUE] = q
    try:
        from logging import getLogger
        getLogger("ipo").info(f"[queue] added item idx={len(q)-1} prompt={prompt!r} steps={steps} size={lstate.width}x{lstate.height}")
    except Exception:
        try:
            print(f"[queue] added item idx={len(q)-1} prompt={prompt!r} steps={steps} size={lstate.width}x{lstate.height}")
        except Exception:
            pass


def _queue_fill_up_to() -> None:
    import streamlit as st
    _ensure_queue()
    raw = getattr(st.session_state, Keys.QUEUE_SIZE, 6)
    size = int(raw if raw is not None else 6)
    import time as _time
    t0 = _time.perf_counter()
    while len(st.session_state.get(Keys.QUEUE) or []) < size:
        _queue_add_one()
    try:
        dt_ms = (_time.perf_counter() - t0) * 1000.0
        from logging import getLogger
        getLogger("ipo").info(f"[queue] filled to size={len(st.session_state.get(Keys.QUEUE) or [])} in {dt_ms:.1f} ms")
    except Exception:
        try:
            print(f"[queue] filled to size={len(st.session_state.get(Keys.QUEUE) or [])} in {dt_ms:.1f} ms")
        except Exception:
            pass


def _queue_label(idx: int, label: int, img=None) -> None:
    import streamlit as st
    from batch_ui import _curation_add, _curation_train_and_next
    _ensure_queue()
    q = st.session_state.get(Keys.QUEUE) or []
    if 0 <= idx < len(q):
        z = q[idx]['z']
        _curation_add(int(label), z, img)
        try:
            _curation_train_and_next()  # retrain from disk; does not alter queue
        except Exception:
            pass
        q.pop(idx)
        st.session_state[Keys.QUEUE] = q
        try:
            from logging import getLogger
            getLogger("ipo").info(f"[queue] labeled idx={idx} label={int(label)} remaining={len(q)}")
        except Exception:
            try:
                print(f"[queue] labeled idx={idx} label={int(label)} remaining={len(q)}")
            except Exception:
                pass


def _render_queue_ui() -> None:
    import streamlit as st
    from value_scorer import get_value_scorer
    from latent_opt import z_from_prompt
    st.subheader("Async queue")
    _ensure_queue()
    q = st.session_state.get(Keys.QUEUE) or []
    if not q:
        st.write("Queue empty…")
    else:
        i = 0
        it = q[0]

        def _render_item() -> None:
            import time as _time

            t0 = _time.perf_counter()
            img = it['future'].result()
            try:
                dt_ms = (_time.perf_counter() - t0) * 1000.0
                from logging import getLogger
                getLogger("ipo").info(f"[queue] decoded and showing item 0 in {dt_ms:.1f} ms")
            except Exception:
                try:
                    print(f"[queue] decoded and showing item 0 in {dt_ms:.1f} ms")
                except Exception:
                    pass
            # Predicted value per queue item
            v_text = "Value: n/a"
            try:
                lstate = st.session_state.lstate
                vm_choice = st.session_state.get('vm_choice')
                scorer = get_value_scorer(vm_choice, lstate, getattr(st.session_state, 'prompt', ''), st.session_state)
                z_p = z_from_prompt(lstate, getattr(st.session_state, 'prompt', ''))
                fvec = (it['z'] - z_p)
                v = float(scorer(fvec))
                v_text = f"Value: {v:.3f}"
            except Exception:
                pass
            st.image(img, caption=f"Item {i}", width="stretch")
            try:
                st.caption(v_text)
            except Exception:
                pass
            if st.button(f"Accept {i}", key=f"queue_accept_{i}", width="stretch"):
                _queue_label(i, 1, img)
            if st.button(f"Reject {i}", key=f"queue_reject_{i}", width="stretch"):
                _queue_label(i, -1, img)

        # Mirror batch_ui: wrap the single visible queue item in a fragment
        # when available so its decode + buttons are scoped to one fragment.
        frag = getattr(st, "fragment", None)
        use_frags = bool(getattr(st.session_state, 'use_fragments', True))
        if use_frags and callable(frag):
            try:
                wrapped = frag(_render_item)
                wrapped()
            except TypeError:
                _render_item()
        else:
            _render_item()
    _queue_fill_up_to()


def run_queue_mode() -> None:
    _ensure_queue()
    _queue_fill_up_to()
    _render_queue_ui()
