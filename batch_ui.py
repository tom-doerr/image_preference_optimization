from __future__ import annotations

import numpy as np


def _lstate_and_prompt():
    import streamlit as st
    return st.session_state.lstate, st.session_state.prompt


def _sample_around_prompt(scale: float = 0.8) -> np.ndarray:
    from latent_logic import z_from_prompt
    lstate, prompt = _lstate_and_prompt()
    z_p = z_from_prompt(lstate, prompt)
    r = lstate.rng.standard_normal(lstate.d)
    r = r / (np.linalg.norm(r) + 1e-12)
    return z_p + lstate.sigma * float(scale) * r


def _curation_init_batch() -> None:
    import streamlit as st
    if st.session_state.get('cur_batch') is None:
        st.session_state.cur_batch = []
        st.session_state.cur_labels = []
        _curation_new_batch()


def _curation_new_batch() -> None:
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    z_list = []
    from latent_logic import z_from_prompt
    z_p = z_from_prompt(lstate, prompt)
    for _ in range(int(st.session_state.get('batch_size', 6))):
        r = lstate.rng.standard_normal(lstate.d)
        r = r / (np.linalg.norm(r) + 1e-12)
        z = z_p + lstate.sigma * 0.8 * r
        z_list.append(z)
    st.session_state.cur_batch = z_list
    st.session_state.cur_labels = [None] * len(z_list)
    st.session_state.batch_futures = [None] * len(z_list)
    st.session_state.batch_started = [None] * len(z_list)


def _curation_replace_at(idx: int) -> None:
    import streamlit as st
    try:
        z_new = _sample_around_prompt(0.8)
        st.session_state.cur_batch[idx] = z_new
        st.session_state.cur_labels[idx] = None
    except Exception:
        pass


def _curation_add(label: int, z: np.ndarray) -> None:
    import streamlit as st
    from persistence import append_dataset_row
    from latent_logic import z_from_prompt
    lstate, prompt = _lstate_and_prompt()
    z_p = z_from_prompt(lstate, prompt)
    X = getattr(st.session_state, 'dataset_X', None)
    y = getattr(st.session_state, 'dataset_y', None)
    feat = (z - z_p).reshape(1, -1)
    lab = np.array([float(label)])
    st.session_state.dataset_X = feat if X is None else np.vstack([X, feat])
    st.session_state.dataset_y = lab if y is None else np.concatenate([y, lab])
    try:
        append_dataset_row(prompt, feat, float(label))
    except Exception:
        pass


def _curation_train_and_next() -> None:
    import streamlit as st
    from persistence import get_dataset_for_prompt_or_session
    from value_model import fit_value_model
    lstate, prompt = _lstate_and_prompt()
    X, y = get_dataset_for_prompt_or_session(prompt, st.session_state)
    if X is not None and y is not None and getattr(X, 'shape', (0,))[0] > 0:
        try:
            lam_now = float(getattr(st.session_state, 'reg_lambda', 1e-3))
            fit_value_model(st.session_state.get('vm_choice'), lstate, X, y, lam_now, st.session_state)
        except Exception:
            pass
    _curation_new_batch()


def _refit_from_dataset_keep_batch() -> None:
    import streamlit as st
    from persistence import get_dataset_for_prompt_or_session
    from value_model import fit_value_model
    lstate, prompt = _lstate_and_prompt()
    X, y = get_dataset_for_prompt_or_session(prompt, st.session_state)
    try:
        if X is not None and y is not None and getattr(X, 'shape', (0,))[0] > 0:
            lam_now = float(getattr(st.session_state, 'reg_lambda', 1e-3))
            fit_value_model(st.session_state.get('vm_choice'), lstate, X, y, lam_now, st.session_state)
    except Exception:
        pass


def _render_batch_ui() -> None:
    import streamlit as st
    import background as bg
    from constants import DECODE_TIMEOUT_S
    from latent_opt import z_to_latents
    from flux_local import generate_flux_image_latents

    st.subheader("Curation batch")
    futs = st.session_state.get('batch_futures') or []
    if not futs or len(futs) != len(st.session_state.cur_batch or []):
        futs = [None] * len(st.session_state.cur_batch or [])
        st.session_state.batch_futures = futs
    starts = st.session_state.get('batch_started') or [None] * len(st.session_state.cur_batch or [])
    if len(starts) != len(st.session_state.cur_batch or []):
        starts = [None] * len(st.session_state.cur_batch or [])
        st.session_state.batch_started = starts

    lstate, prompt = _lstate_and_prompt()
    steps = int(getattr(st.session_state, 'steps', 6))
    guidance_eff = float(getattr(st.session_state, 'guidance_eff', 0.0))

    for i, z_i in enumerate(st.session_state.cur_batch or []):
        if futs[i] is None:
            la = z_to_latents(lstate, z_i)
            futs[i] = bg.schedule_decode_latents(prompt, la, lstate.width, lstate.height, steps, guidance_eff)
            st.session_state.batch_futures = futs
            import time as _time
            starts[i] = _time.time()
            st.session_state.batch_started = starts

        def _sync_decode():
            la2 = z_to_latents(lstate, z_i)
            return generate_flux_image_latents(prompt, latents=la2, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff)

        img_i, futs[i] = bg.result_or_sync_after(futs[i], starts[i], DECODE_TIMEOUT_S, _sync_decode)
        st.session_state.batch_futures = futs

        if img_i is not None:
            st.image(img_i, caption=f"Item {i}", use_container_width=True)
        else:
            st.write(f"Item {i}: loading…")
            continue

        if st.button(f"Good (+1) {i}", use_container_width=True):
            import time as _time
            t0 = _time.perf_counter()
            _curation_add(1, z_i)
            st.session_state.cur_labels[i] = 1
            _refit_from_dataset_keep_batch()
            _curation_replace_at(i)
            try:
                print(f"[perf] good_label item={i} took {(_time.perf_counter()-t0)*1000:.1f} ms")
            except Exception:
                pass
            if callable(getattr(st, 'rerun', None)):
                try:
                    st.rerun()
                except Exception:
                    pass
        if st.button(f"Bad (-1) {i}", use_container_width=True):
            import time as _time
            t0 = _time.perf_counter()
            _curation_add(-1, z_i)
            st.session_state.cur_labels[i] = -1
            _refit_from_dataset_keep_batch()
            _curation_replace_at(i)
            try:
                print(f"[perf] bad_label item={i} took {(_time.perf_counter()-t0)*1000:.1f} ms")
            except Exception:
                pass
            if callable(getattr(st, 'rerun', None)):
                try:
                    st.rerun()
                except Exception:
                    pass


def run_batch_mode() -> None:
    _curation_init_batch()
    _render_batch_ui()

