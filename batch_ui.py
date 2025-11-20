from __future__ import annotations

from typing import Any, Tuple
import numpy as np
from constants import Keys
import logging as _logging

LOGGER = _logging.getLogger("ipo")
if not LOGGER.handlers:
    try:
        _h = _logging.FileHandler("ipo.debug.log")
        _h.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s batch_ui: %(message)s"))
        LOGGER.addHandler(_h)
        LOGGER.setLevel(_logging.INFO)
    except Exception:
        pass
try:
    import os as _os
    _lvl = (_os.getenv("IPO_LOG_LEVEL") or "").upper()
    if _lvl:
        LOGGER.setLevel(getattr(_logging, _lvl, _logging.INFO))
except Exception:
    pass

def _log(msg: str, level: str = "info") -> None:
    try:
        print(msg)
    except Exception:
        pass
    try:
        getattr(LOGGER, level, LOGGER.info)(msg)
    except Exception:
        pass

__all__ = [
    '_lstate_and_prompt',
    '_sample_around_prompt',
    '_curation_init_batch',
    '_curation_new_batch',
    '_curation_replace_at',
    '_curation_add',
    '_curation_train_and_next',
    '_refit_from_dataset_keep_batch',
    '_render_batch_ui',
    'run_batch_mode',
]


def _lstate_and_prompt() -> Tuple[Any, str]:
    import streamlit as st
    lstate = getattr(st.session_state, 'lstate', None)
    if lstate is None:
        try:
            from latent_state import init_latent_state as _init
            lstate = _init()
            st.session_state.lstate = lstate
        except Exception:
            pass
    prompt = getattr(st.session_state, 'prompt', None)
    if not prompt:
        from constants import DEFAULT_PROMPT
        prompt = DEFAULT_PROMPT
    return lstate, prompt


def _sample_around_prompt(scale: float = 0.8) -> np.ndarray:
    from latent_logic import z_from_prompt
    lstate, prompt = _lstate_and_prompt()
    z_p = z_from_prompt(lstate, prompt)
    r = lstate.rng.standard_normal(lstate.d)
    r = r / (np.linalg.norm(r) + 1e-12)
    z = z_p + lstate.sigma * float(scale) * r
    _log(f"[latent] sample_around_prompt scale={scale} ‖z_p‖={float(np.linalg.norm(z_p)):.3f} ‖z‖={float(np.linalg.norm(z)):.3f}")
    return z


def _curation_init_batch() -> None:
    # Always create a fresh batch on init so each page reload/new round uses
    # newly sampled latents instead of reusing the previous cur_batch.
    _curation_new_batch()


def _curation_new_batch() -> None:
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    import time as _time
    t0 = _time.perf_counter()
    z_list = []
    from latent_logic import z_from_prompt, sample_z_xgb_hill
    z_p = z_from_prompt(lstate, prompt)
    batch_n = int(st.session_state.get('batch_size', 6))
    # Optional XGBoost-guided hill climb per image when XGB is active.
    vm_choice = str(st.session_state.get(Keys.VM_CHOICE) or "")
    use_xgb = (vm_choice == "XGBoost")
    scorer = None
    scorer_status = None
    steps = int(st.session_state.get(Keys.ITER_STEPS, 10))
    lr_mu = float(st.session_state.get(Keys.LR_MU_UI, 0.3))
    trust = st.session_state.get(Keys.TRUST_R, None)
    trust_r = float(trust) if (trust is not None and float(trust) > 0.0) else None
    if use_xgb:
        try:
            from persistence import get_dataset_for_prompt_or_session
            from value_model import ensure_fitted
            X_ds, y_ds = get_dataset_for_prompt_or_session(prompt, st.session_state)
            if X_ds is not None and y_ds is not None and getattr(X_ds, "shape", (0,))[0] > 0:
                try:
                    d_x = int(getattr(X_ds, "shape", (0, 0))[1])
                    d_lat = int(getattr(lstate, "d", d_x))
                    if d_x == d_lat:
                        lam_now = float(st.session_state.get(Keys.REG_LAMBDA, 1e-3))
                        vm_train_choice = str(st.session_state.get("vm_train_choice", vm_choice))
                        ensure_fitted(vm_train_choice, lstate, X_ds, y_ds, lam_now, st.session_state)
                except Exception:
                    pass
            from value_scorer import get_value_scorer_with_status
            scorer, scorer_status = get_value_scorer_with_status("XGBoost", lstate, prompt, st.session_state)
            if scorer_status != "ok":
                scorer = None
        except Exception:
            scorer = None
    for i in range(batch_n):
        z = None
        if use_xgb and scorer is not None:
            try:
                step_scale = lr_mu * float(getattr(lstate, "sigma", 1.0))
                z = sample_z_xgb_hill(lstate, prompt, scorer, steps=steps, step_scale=step_scale, trust_r=trust_r)
            except Exception:
                z = None
        if z is None:
            r = lstate.rng.standard_normal(lstate.d)
            r = r / (np.linalg.norm(r) + 1e-12)
            z = z_p + lstate.sigma * 0.8 * r
        z_list.append(z)
    st.session_state.cur_batch = z_list
    st.session_state.cur_labels = [None] * len(z_list)
    try:
        st.session_state["cur_batch_nonce"] = int(st.session_state.get("cur_batch_nonce", 0)) + 1
    except Exception:
        pass
    try:
        dt_ms = (_time.perf_counter() - t0) * 1000.0
    except Exception:
        dt_ms = -1.0
    _log(f"[batch] new batch: n={len(z_list)} d={lstate.d} sigma={lstate.sigma:.3f} ‖z_p‖={float(np.linalg.norm(z_p)):.3f} size={lstate.width}x{lstate.height} in {dt_ms:.1f} ms")


def _curation_replace_at(idx: int) -> None:
    # After each label we refresh the whole batch; the index is unused but
    # kept for test compatibility.
    try:
        _curation_new_batch()
    except Exception:
        pass


def _curation_add(label: int, z: np.ndarray, img=None) -> None:
    import streamlit as st
    from persistence import append_dataset_row, save_sample_image
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
        row_idx = append_dataset_row(prompt, feat, float(label))
        if img is not None:
            save_sample_image(prompt, row_idx, img)
        try:
            getattr(st, "toast", lambda *a, **k: None)(f"Saved sample #{row_idx}")
        except Exception:
            pass
    except Exception:
        pass


def _curation_train_and_next() -> None:
    import streamlit as st
    from persistence import get_dataset_for_prompt_or_session
    from value_model import ensure_fitted
    lstate, prompt = _lstate_and_prompt()
    # Respect toggle: skip training when disabled
    if not bool(st.session_state.get('train_on_new_data', True)):
        _curation_new_batch()
        return
    # Track async XGB training status in session for UI/reruns
    st.session_state.pop(Keys.XGB_FIT_FUTURE, None)
    st.session_state.pop(Keys.XGB_TRAIN_STATUS, None)
    X, y = get_dataset_for_prompt_or_session(prompt, st.session_state)
    # persistence.get_dataset_for_prompt_or_session already guards dim mismatches
    if X is not None and y is not None and getattr(X, 'shape', (0,))[0] > 0:
        try:
            lam_now = float(getattr(st.session_state, Keys.REG_LAMBDA, 1e-3))
            vm_train = str(st.session_state.get(Keys.VM_CHOICE))
            ensure_fitted(vm_train, lstate, X, y, lam_now, st.session_state)
            try:
                getattr(st, "toast", lambda *a, **k: None)(f"Training {vm_train}…")
            except Exception:
                pass
            import value_model as _vm
            # Ensure status is visible immediately for tests/UI
            try:
                st.session_state[Keys.XGB_TRAIN_STATUS] = {"state": "running", "rows": int(getattr(X, 'shape', (0,))[0]), "lam": float(lam_now)}
            except Exception:
                pass
            tr = getattr(_vm, 'train_and_record', None)
            if callable(tr):
                tr(vm_train, lstate, X, y, lam_now, st.session_state)
            else:
                _vm.fit_value_model(vm_train, lstate, X, y, lam_now, st.session_state)
        except Exception:
            pass
    _curation_new_batch()


def _refit_from_dataset_keep_batch() -> None:
    import streamlit as st
    from persistence import get_dataset_for_prompt_or_session
    from value_model import ensure_fitted
    lstate, prompt = _lstate_and_prompt()
    if not bool(st.session_state.get('train_on_new_data', True)):
        return
    st.session_state.pop("xgb_fit_future", None)
    st.session_state.pop("xgb_train_status", None)
    X, y = get_dataset_for_prompt_or_session(prompt, st.session_state)
    # persistence.get_dataset_for_prompt_or_session already guards dim mismatches
    try:
        if X is not None and y is not None and getattr(X, 'shape', (0,))[0] > 0:
            lam_now = float(getattr(st.session_state, Keys.REG_LAMBDA, 1e-3))
            vm_train = str(st.session_state.get(Keys.VM_CHOICE))
            ensure_fitted(vm_train, lstate, X, y, lam_now, st.session_state)
            try:
                getattr(st, "toast", lambda *a, **k: None)(f"Training {vm_train}…")
            except Exception:
                pass
            import value_model as _vm
            try:
                st.session_state[Keys.XGB_TRAIN_STATUS] = {"state": "running", "rows": int(getattr(X, 'shape', (0,))[0]), "lam": float(lam_now)}
            except Exception:
                pass
            tr = getattr(_vm, 'train_and_record', None)
            if callable(tr):
                tr(vm_train, lstate, X, y, lam_now, st.session_state)
            else:
                _vm.fit_value_model(vm_train, lstate, X, y, lam_now, st.session_state)
    except Exception:
        pass


def _render_batch_ui() -> None:
    import streamlit as st
    from latent_logic import z_to_latents, z_from_prompt
    from flux_local import generate_flux_image_latents
    import time as _time

    (getattr(st, 'subheader', lambda *a, **k: None))("Curation batch")
    # Reset per-render button sequence to keep keys unique yet bounded
    try:
        st.session_state["btn_seq"] = 0
    except Exception:
        pass
    try:
        st.session_state["render_nonce"] = int(st.session_state.get("render_nonce", 0)) + 1
    except Exception:
        pass
    render_nonce = int(st.session_state.get("render_nonce", 0))
    lstate, prompt = _lstate_and_prompt()
    steps = int(getattr(st.session_state, 'steps', 6))
    guidance_eff = float(getattr(st.session_state, 'guidance_eff', 0.0))
    best_of = bool(getattr(st.session_state, 'batch_best_of', False))
    cur_batch = st.session_state.cur_batch or []
    # Prepare optional value scorer once per batch
    scorer = None
    scorer_status = None
    fut_running = False
    z_p = None
    try:
        vm_choice = st.session_state.get('vm_choice')
        from value_scorer import get_value_scorer_with_status
        scorer, scorer_status = get_value_scorer_with_status(vm_choice, lstate, prompt, st.session_state)
        fut = st.session_state.get("xgb_fit_future")
        fut_running = bool(fut is not None and not getattr(fut, "done", lambda: False)())
        if scorer_status != "ok":
            scorer = None
        z_p = z_from_prompt(lstate, prompt)
    except Exception:
        scorer = None
    n = len(cur_batch)
    if n == 0:
        return
    per_row = min(5, n)

    for row_start in range(0, n, per_row):
        row_end = min(row_start + per_row, n)
        cols = getattr(st, 'columns', lambda x: [None] * x)(row_end - row_start)
        for col_idx, i in enumerate(range(row_start, row_end)):
            col = cols[col_idx] if cols and len(cols) > col_idx else None

            def _render_item() -> None:
                # Create the vector for this image immediately before decode so
                # each tile uses a freshly sampled latent under the current
                # settings, independent of cur_batch.
                z_i = cur_batch[i]
                try:
                    vm_choice_local = st.session_state.get('vm_choice')
                except Exception:
                    vm_choice_local = None
                if vm_choice_local == "XGBoost" and scorer is not None and not fut_running:
                    try:
                        from latent_logic import sample_z_xgb_hill  # local import
                        steps_local = int(st.session_state.get('iter_steps', 10))
                        lr_mu_local = float(st.session_state.get('lr_mu_ui', 0.3))
                        trust_val = st.session_state.get('trust_r', None)
                        trust_r_local = float(trust_val) if (trust_val is not None and float(trust_val) > 0.0) else None
                        step_scale_local = lr_mu_local * float(getattr(lstate, "sigma", 1.0))
                        z_i = sample_z_xgb_hill(
                            lstate,
                            prompt,
                            scorer,
                            steps=steps_local,
                            step_scale=step_scale_local,
                            trust_r=trust_r_local,
                        )
                    except Exception:
                        pass
                else:
                    # Non-XGB modes: reuse the shared helper to keep logic DRY.
                    try:
                        z_i = _sample_around_prompt(scale=0.8)
                    except Exception:
                        z_i = cur_batch[i]

                # Persist the per-image latent so reruns see the same vector
                # until a new batch is created.
                try:
                    cur_batch[i] = z_i
                    st.session_state.cur_batch = cur_batch
                except Exception:
                    pass

                t0 = _time.perf_counter()
                la = z_to_latents(lstate, z_i)
                img_i = generate_flux_image_latents(
                    prompt,
                    latents=la,
                    width=lstate.width,
                    height=lstate.height,
                    steps=steps,
                    guidance=guidance_eff,
                )
                try:
                    dt_ms = (_time.perf_counter() - t0) * 1000.0
                except Exception:
                    dt_ms = -1.0
                _log(f"[batch] decoded item={i} in {dt_ms:.1f} ms (steps={steps}, w={lstate.width}, h={lstate.height})")
                # Predicted value using current value model scorer
                v_text = "Value: n/a"
                if scorer is not None and z_p is not None:
                    try:
                        fvec = (z_i - z_p)
                        v = float(scorer(fvec))
                        v_text = f"Value: {v:.3f}"
                    except Exception:
                        v_text = "Value: n/a"
                st.image(img_i, caption=f"Item {i}", width="stretch")
                try:
                    st.caption(v_text)
                except Exception:
                    pass

                if best_of:
                    if st.button(f"Choose {i}", key=f"choose_{i}", width="stretch"):
                        t0b = _time.perf_counter()
                        for j, z_j in enumerate(cur_batch):
                            lbl = 1 if j == i else -1
                            # Only have img_i for the chosen index; others save features only.
                            img_j = img_i if j == i else None
                            _curation_add(lbl, z_j, img_j)
                            st.session_state.cur_labels[j] = lbl
                        _curation_train_and_next()
                        try:
                            getattr(st, "toast", lambda *a, **k: None)(f"Best-of: chose {i}")
                        except Exception:
                            pass
                        _log(f"[perf] best_of choose item={i} took {(_time.perf_counter()-t0b)*1000:.1f} ms")
                else:
                    btn_cols = getattr(st, "columns", lambda x: [None] * x)(2)
                    gcol = btn_cols[0] if btn_cols and len(btn_cols) > 0 else None
                    bcol = btn_cols[1] if btn_cols and len(btn_cols) > 1 else None

                    nonce = int(st.session_state.get("cur_batch_nonce", 0))

                    def _btn_key(prefix: str, idx: int) -> str:
                        try:
                            seq = int(st.session_state.get("btn_seq", 0)) + 1
                        except Exception:
                            seq = 1
                        st.session_state["btn_seq"] = seq
                        return f"{prefix}_{render_nonce}_{nonce}_{idx}_{seq}"

                    def _good_clicked() -> bool:
                        if gcol is not None:
                            with gcol:
                                return st.button(f"Good (+1) {i}", key=_btn_key("good", i), width="stretch")
                        return st.button(f"Good (+1) {i}", key=_btn_key("good", i), width="stretch")

                    def _bad_clicked() -> bool:
                        if bcol is not None:
                            with bcol:
                                return st.button(f"Bad (-1) {i}", key=_btn_key("bad", i), width="stretch")
                        return st.button(f"Bad (-1) {i}", key=_btn_key("bad", i), width="stretch")

                    if _good_clicked():
                        t0g = _time.perf_counter()
                        _curation_add(1, z_i, img_i)
                        st.session_state.cur_labels[i] = 1
                        _refit_from_dataset_keep_batch()
                        _curation_replace_at(i)
                        try:
                            getattr(st, "toast", lambda *a, **k: None)("Labeled Good (+1)")
                        except Exception:
                            pass
                        _log(f"[perf] good_label item={i} took {(_time.perf_counter()-t0g)*1000:.1f} ms")
                    if _bad_clicked():
                        t0b2 = _time.perf_counter()
                        _curation_add(-1, z_i, img_i)
                        st.session_state.cur_labels[i] = -1
                        _refit_from_dataset_keep_batch()
                        _curation_replace_at(i)
                        try:
                            getattr(st, "toast", lambda *a, **k: None)("Labeled Bad (-1)")
                        except Exception:
                            pass
                        _log(f"[perf] bad_label item={i} took {(_time.perf_counter()-t0b2)*1000:.1f} ms")

            # Wrap each tile in its own fragment when available so the
            # latent sampling, decode, buttons, and saves are scoped per
            # image and can run independently. Streamlit exposes fragments
            # as a decorator, so we decorate _render_item and then call it.
            frag = getattr(st, "fragment", None)
            use_frags = bool(getattr(st.session_state, 'use_fragments', True))
            if col is not None:
                with col:
                    if use_frags and callable(frag):
                        try:
                            wrapped = frag(_render_item)
                            wrapped()
                        except TypeError:
                            _render_item()
                    else:
                        _render_item()
            else:
                if use_frags and callable(frag):
                    try:
                        wrapped = frag(_render_item)
                        wrapped()
                    except TypeError:
                        _render_item()
                else:
                    _render_item()


def run_batch_mode() -> None:
    _curation_init_batch()
    _render_batch_ui()
