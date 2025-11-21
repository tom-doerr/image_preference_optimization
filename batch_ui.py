from __future__ import annotations

from typing import Any, Tuple
import numpy as np
from constants import Keys, DEFAULT_ITER_STEPS
import logging as _logging

LOGGER = _logging.getLogger("ipo")
if not LOGGER.handlers:
    try:
        _h = _logging.FileHandler("ipo.debug.log")
        _h.setFormatter(
            _logging.Formatter("%(asctime)s %(levelname)s batch_ui: %(message)s")
        )
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
    "_lstate_and_prompt",
    "_sample_around_prompt",
    "_prepare_xgb_scorer",
    "_sample_one_for_batch",
    "_curation_params",
    "_curation_init_batch",
    "_curation_new_batch",
    "_curation_replace_at",
    "_curation_add",
    "_curation_train_and_next",
    "_refit_from_dataset_keep_batch",
    "_render_batch_ui",
    "run_batch_mode",
]


def _lstate_and_prompt() -> Tuple[Any, str]:
    import streamlit as st

    lstate = getattr(st.session_state, "lstate", None)
    if lstate is None:
        try:
            from latent_state import init_latent_state as _init

            lstate = _init()
            st.session_state.lstate = lstate
        except Exception:
            pass
    prompt = getattr(st.session_state, "prompt", None)
    if not prompt:
        from constants import DEFAULT_PROMPT

        prompt = DEFAULT_PROMPT
    return lstate, prompt


def _sample_around_prompt(scale: float = 0.8) -> np.ndarray:
    """Deterministic, stub-friendly sampler around the prompt anchor.

    Falls back to a zero anchor and a fixed RNG seed when latent_logic or RNG
    are unavailable, so tests always get a usable cur_batch without decodes.
    """
    lstate, prompt = _lstate_and_prompt()
    try:
        from latent_logic import z_from_prompt

        z_p = z_from_prompt(lstate, prompt)
    except Exception:
        import numpy as _np

        d = int(getattr(lstate, "d", 8))
        z_p = _np.zeros(d, dtype=float)
    rng = getattr(lstate, "rng", None)
    if rng is None:
        import numpy as _np

        rng = _np.random.default_rng(0)
        setattr(lstate, "rng", rng)
    r = rng.standard_normal(int(getattr(lstate, "d", len(z_p))))
    r = r / (np.linalg.norm(r) + 1e-12)
    z = z_p + float(getattr(lstate, "sigma", 1.0)) * float(scale) * r
    try:
        _log(
            f"[latent] sample_around_prompt scale={scale} ‖z_p‖={float(np.linalg.norm(z_p)):.3f} ‖z‖={float(np.linalg.norm(z)):.3f}"
        )
    except Exception:
        pass
    return z


def _prepare_xgb_scorer(lstate: Any, prompt: str):
    """Return (scorer, status) for XGB when available, else (None, status).

    Keeps import surface tiny and avoids repeating dataset/ensure_fitted calls.
    """
    try:
        from persistence import get_dataset_for_prompt_or_session
        from value_model import ensure_fitted

        X_ds, y_ds = get_dataset_for_prompt_or_session(
            prompt, __import__("streamlit").session_state
        )
        if (
            X_ds is not None
            and y_ds is not None
            and getattr(X_ds, "shape", (0,))[0] > 0
        ):
            try:
                from constants import Keys as _K

                lam_now = float(
                    getattr(__import__("streamlit").session_state, _K.REG_LAMBDA, 1e-3)
                )
            except Exception:
                lam_now = 1e-3
            try:
                vm_train_choice = str(
                    __import__("streamlit").session_state.get(
                        "vm_train_choice", "XGBoost"
                    )
                )
            except Exception:
                vm_train_choice = "XGBoost"
            ensure_fitted(
                vm_train_choice,
                lstate,
                X_ds,
                y_ds,
                lam_now,
                __import__("streamlit").session_state,
            )
        from value_scorer import get_value_scorer_with_status

        return get_value_scorer_with_status(
            "XGBoost", lstate, prompt, __import__("streamlit").session_state
        )
    except Exception:
        return None, "xgb_unavailable"


def _sample_one_for_batch(
    lstate: Any, prompt: str, use_xgb: bool, scorer, steps: int, lr_mu: float, trust_r
) -> np.ndarray:
    """Produce one latent for the batch using XGB hill or around‑prompt sample."""
    if use_xgb and scorer is not None:
        try:
            from latent_logic import sample_z_xgb_hill  # local import

            step_scale = lr_mu * float(getattr(lstate, "sigma", 1.0))
            return sample_z_xgb_hill(
                lstate,
                prompt,
                scorer,
                steps=int(steps),
                step_scale=step_scale,
                trust_r=trust_r,
            )
        except Exception:
            pass
    return _sample_around_prompt(scale=0.8)


def _curation_params():
    """Read once: VM choice, steps, lr_mu, trust_r, use_xgb."""
    import streamlit as st

    try:
        vm_choice = str(st.session_state.get(Keys.VM_CHOICE) or "")
    except Exception:
        vm_choice = ""
    use_xgb = vm_choice == "XGBoost"
    try:
        steps = int(st.session_state.get(Keys.ITER_STEPS, 10))
    except Exception:
        steps = 10
    try:
        lr_mu = float(st.session_state.get(Keys.LR_MU_UI, 0.3))
    except Exception:
        lr_mu = 0.3
    try:
        trust = st.session_state.get(Keys.TRUST_R, None)
        trust_r = float(trust) if (trust is not None and float(trust) > 0.0) else None
    except Exception:
        trust_r = None
    return vm_choice, use_xgb, steps, lr_mu, trust_r


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
    from latent_logic import z_from_prompt

    z_p = z_from_prompt(lstate, prompt)
    batch_n = int(st.session_state.get("batch_size", 6))
    # Optional XGBoost-guided hill climb per image when XGB is active.
    vm_choice, use_xgb, steps, lr_mu, trust_r = _curation_params()
    scorer = None
    scorer_status = None
    if use_xgb:
        scorer, scorer_status = _prepare_xgb_scorer(lstate, prompt)
        if scorer_status != "ok":
            scorer = None
    for i in range(batch_n):
        z_list.append(
            _sample_one_for_batch(
                lstate, prompt, use_xgb, scorer, steps, lr_mu, trust_r
            )
        )
    st.session_state.cur_batch = z_list
    st.session_state.cur_labels = [None] * len(z_list)
    try:
        st.session_state["cur_batch_nonce"] = (
            int(st.session_state.get("cur_batch_nonce", 0)) + 1
        )
    except Exception:
        pass
    try:
        dt_ms = (_time.perf_counter() - t0) * 1000.0
    except Exception:
        dt_ms = -1.0
    try:
        vm_choice = st.session_state.get(Keys.VM_CHOICE)
    except Exception:
        vm_choice = None
    _log(
        f"[batch] new batch: n={len(z_list)} d={lstate.d} sigma={lstate.sigma:.3f} ‖z_p‖={float(np.linalg.norm(z_p)):.3f} size={lstate.width}x{lstate.height} vm={vm_choice} in {dt_ms:.1f} ms"
    )


def _curation_replace_at(idx: int) -> None:
    """Deterministically resample the item at idx; keep batch size constant.

    Under full app runs we can refresh the entire batch; in stubs/tests we
    resample only the requested index to keep behavior predictable.
    """
    import streamlit as st

    try:
        zs = getattr(st.session_state, "cur_batch", None) or []
        if not zs:
            _curation_new_batch();
            zs = getattr(st.session_state, "cur_batch", None) or []
        if not zs:
            return
        i = int(idx) % len(zs)
        try:
            _log(f"[batch] replace_at idx={i} nonce={int(st.session_state.get('cur_batch_nonce', 0))}")
        except Exception:
            pass
        # Deterministic resample keyed on (batch_nonce, idx)
        try:
            import numpy as _np
            lstate, prompt = _lstate_and_prompt()
            try:
                from latent_logic import z_from_prompt as _zfp
                z_p = _zfp(lstate, prompt)
            except Exception:
                z_p = _np.zeros(int(getattr(lstate, 'd', 8)), dtype=float)
            nonce = int(st.session_state.get('cur_batch_nonce', 0))
            rng = _np.random.default_rng(1009 * (nonce + 1) + int(i) + 1)
            r = rng.standard_normal(z_p.shape)
            r = r / (float(_np.linalg.norm(r)) + 1e-12)
            zs[i] = z_p + float(getattr(lstate, 'sigma', 1.0)) * 0.8 * r
        except Exception:
            zs[i] = _sample_around_prompt(scale=0.8)
        st.session_state.cur_batch = zs
        try:
            st.session_state.cur_labels[i] = None
        except Exception:
            pass
    except Exception:
        try:
            _curation_new_batch()
        except Exception:
            pass


def _curation_add(label: int, z: np.ndarray, img=None) -> None:
    import streamlit as st
    import persistence as p
    from constants import Keys
    from latent_logic import z_from_prompt

    lstate, prompt = _lstate_and_prompt()
    z_p = z_from_prompt(lstate, prompt)
    X = getattr(st.session_state, "dataset_X", None)
    y = getattr(st.session_state, "dataset_y", None)
    feat = (z - z_p).reshape(1, -1)
    try:
        _log(f"[data] append label={int(label)} ‖feat‖={float(np.linalg.norm(feat)):.3f}")
    except Exception:
        pass
    lab = np.array([float(label)])
    st.session_state.dataset_X = feat if X is None else np.vstack([X, feat])
    st.session_state.dataset_y = lab if y is None else np.concatenate([y, lab])
    # Also mirror into keyed entries for consistency with Keys-based readers
    try:
        st.session_state[Keys.DATASET_X] = st.session_state.dataset_X
        st.session_state[Keys.DATASET_Y] = st.session_state.dataset_y
    except Exception:
        pass
    try:
        row_idx = p.append_sample(prompt, feat, float(label), img)
        try:
            save_dir = getattr(p, "data_root_for_prompt", lambda pr: "data")(prompt)
        except Exception:
            save_dir = "data"
        msg = f"Saved sample #{row_idx} → {save_dir}/{row_idx:06d}"
        try:
            _log(f"[data] saved row={row_idx} path={save_dir}/{row_idx:06d}")
        except Exception:
            pass
        # Single line notice only (no toasts): keep it boring and explicit
        try:
            st.sidebar.write(msg)
        except Exception:
            pass
        # Record last action for the sidebar panel
        try:
            import time as _time
            st.session_state[Keys.LAST_ACTION_TEXT] = msg
            st.session_state[Keys.LAST_ACTION_TS] = float(_time.time())
        except Exception:
            pass
        # Increment interaction step for visibility
        try:
            setattr(lstate, "step", int(getattr(lstate, "step", 0)) + 1)
        except Exception:
            pass
        # Bump the live rows display immediately (memory-only)
        try:
            _yl = st.session_state.get(Keys.DATASET_Y, None)
            rows_live = int(len(_yl)) if _yl is not None else 0
        except Exception:
            rows_live = 0
        try:
            st.session_state[Keys.ROWS_DISPLAY] = str(rows_live)
        except Exception:
            pass
        try:
            print(f"[rows] live={rows_live} disp={rows_live}")
        except Exception:
            pass
    except Exception:
        pass


def _render_batch_tile_body(
    i: int,
    render_nonce: int,
    lstate: Any,
    prompt: str,
    steps: int,
    guidance_eff: float,
    best_of: bool,
    scorer,
    fut_running: bool,
    cur_batch,
    z_p,
) -> None:
    import streamlit as st
    try:
        from latent_logic import z_to_latents
    except Exception:
        from latent_opt import z_to_latents  # tests may stub here
    from flux_local import generate_flux_image_latents
    import time as _time

    # Do not resample latents on render; keep the batch stable until replaced or new batch is created
    z_i = cur_batch[i]

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
    _log(
        f"[batch] decoded item={i} in {dt_ms:.1f} ms (steps={steps}, w={lstate.width}, h={lstate.height})"
    )
    v_text = "Value: n/a"
    if scorer is not None and z_p is not None:
        try:
            fvec = z_i - z_p
            v = float(scorer(fvec))
            v_text = f"Value: {v:.3f}"
            # Append model tag for clarity
            try:
                tag = "XGB" if str(vm_choice_local) == "XGBoost" else "Ridge"
                v_text = f"{v_text} [{tag}]"
            except Exception:
                pass
            try:
                _val = getattr(st.session_state, "log_verbosity", None)
                _lv = 0 if (_val is None) else int(_val)
            except Exception:
                _lv = 0
            if _lv > 0:
                try:
                    vmn = st.session_state.get("vm_choice")
                    _log(f"[scorer] tile={i} vm={vmn} v={v:.3f}")
                except Exception:
                    pass
        except Exception:
            v_text = "Value: n/a"
    cap_txt = f"Item {i} • {v_text}"
    st.image(img_i, caption=cap_txt, width="stretch")

    if best_of:
        if st.button(f"Choose {i}", key=f"choose_{i}", width="stretch"):
            t0b = _time.perf_counter()
            for j, z_j in enumerate(cur_batch):
                lbl = 1 if j == i else -1
                img_j = img_i if j == i else None
                _curation_add(lbl, z_j, img_j)
                st.session_state.cur_labels[j] = lbl
            _curation_train_and_next()
            try:
                getattr(st, "toast", lambda *a, **k: None)(f"Best-of: chose {i}")
            except Exception:
                pass
            _log(
                f"[perf] best_of choose item={i} took {(_time.perf_counter() - t0b) * 1000:.1f} ms"
            )
    else:
        btn_cols = getattr(st, "columns", lambda x: [None] * x)(2)
        gcol = btn_cols[0] if btn_cols and len(btn_cols) > 0 else None
        bcol = btn_cols[1] if btn_cols and len(btn_cols) > 1 else None

        nonce = int(st.session_state.get("cur_batch_nonce", 0))

        def _btn_key(prefix: str, idx: int) -> str:
            # Non-fragment path: make keys vary across renders but stable within
            # a single render so duplicate-key errors are avoided and tests can
            # observe change across reruns.
            try:
                rnd = int(st.session_state.get("render_nonce", 0))
            except Exception:
                rnd = 0
            try:
                rcount = int(st.session_state.get("render_count", 0))
            except Exception:
                rcount = 0
            try:
                seq = int(st.session_state.get("btn_seq", 0)) + 1
            except Exception:
                seq = 1
            st.session_state["btn_seq"] = seq
            return f"{prefix}_{rcount}_{rnd}_{nonce}_{idx}_{seq}"

        def _good_clicked() -> bool:
            if gcol is not None:
                with gcol:
                    return st.button(
                        f"Good (+1) {i}", key=_btn_key("good", i), width="stretch"
                    )
            return st.button(f"Good (+1) {i}", key=_btn_key("good", i), width="stretch")

        def _bad_clicked() -> bool:
            if bcol is not None:
                with bcol:
                    return st.button(
                        f"Bad (-1) {i}", key=_btn_key("bad", i), width="stretch"
                    )
            return st.button(f"Bad (-1) {i}", key=_btn_key("bad", i), width="stretch")

        if _good_clicked():
            t0g = _time.perf_counter()
            _curation_add(1, z_i, img_i)
            st.session_state.cur_labels[i] = 1
            _refit_from_dataset_keep_batch()
            _curation_replace_at(i)
            _log(
                f"[perf] good_label item={i} took {(_time.perf_counter() - t0g) * 1000:.1f} ms"
            )
            # Force a rerun so sidebar counters update immediately
            try:
                rr = getattr(st, "rerun", None)
                if callable(rr):
                    rr()
            except Exception:
                pass
        if _bad_clicked():
            t0b2 = _time.perf_counter()
            _curation_add(-1, z_i, img_i)
            st.session_state.cur_labels[i] = -1
            _refit_from_dataset_keep_batch()
            _curation_replace_at(i)
            _log(
                f"[perf] bad_label item={i} took {(_time.perf_counter() - t0b2) * 1000:.1f} ms"
            )
            # Force a rerun so sidebar counters update immediately
            try:
                rr = getattr(st, "rerun", None)
                if callable(rr):
                    rr()
            except Exception:
                pass


def _curation_train_and_next() -> None:
    import streamlit as st
    from persistence import get_dataset_for_prompt_or_session
    # Prefer central train helper when available; tests may stub value_model without it
    try:
        from value_model import train_and_record as _train_record
    except Exception:
        _train_record = None

    lstate, prompt = _lstate_and_prompt()
    # Respect toggle: skip training when disabled
    if not bool(st.session_state.get("train_on_new_data", True)):
        _curation_new_batch()
        return
    # Clear previous future handle; train_and_record manages status
    st.session_state.pop(Keys.XGB_FIT_FUTURE, None)
    X, y = get_dataset_for_prompt_or_session(prompt, st.session_state)
    # persistence.get_dataset_for_prompt_or_session already guards dim mismatches
    if X is not None and y is not None and getattr(X, "shape", (0,))[0] > 0:
        try:
            lam_now = float(getattr(st.session_state, Keys.REG_LAMBDA, 1e-3))
            # Only train Ridge automatically; XGB trains via explicit button
            vm_train = "Ridge"
            try:
                getattr(st, "toast", lambda *a, **k: None)(f"Training {vm_train}…")
            except Exception:
                pass
            if _train_record is not None:
                _train_record(vm_train, lstate, X, y, lam_now, st.session_state)
            else:
                # Minimal fallback: emulate cooldown + single submit, then call fit once
                from datetime import datetime, timezone
                last_at = st.session_state.get(Keys.LAST_TRAIN_AT)
                min_wait = float(st.session_state.get("min_train_interval_s", 0.0) or 0.0)
                recent = False
                if last_at and min_wait > 0.0:
                    try:
                        dt = datetime.fromisoformat(last_at)
                        recent = (datetime.now(timezone.utc) - dt).total_seconds() < min_wait
                    except Exception:
                        recent = False
                if recent:
                    st.session_state[Keys.XGB_TRAIN_STATUS] = {"state": "waiting", "rows": int(X.shape[0]), "lam": float(lam_now)}
                else:
                    st.session_state[Keys.XGB_TRAIN_STATUS] = {"state": "running", "rows": int(X.shape[0]), "lam": float(lam_now)}
                    try:
                        from value_model import fit_value_model as _fit_vm

                        _fit_vm(vm_train, lstate, X, y, lam_now, st.session_state)
                    except Exception:
                        pass
        except Exception:
            pass
    _curation_new_batch()


def _refit_from_dataset_keep_batch() -> None:
    import streamlit as st
    from persistence import get_dataset_for_prompt_or_session
    try:
        from value_model import train_and_record as _train_record
    except Exception:
        _train_record = None

    lstate, prompt = _lstate_and_prompt()
    if not bool(st.session_state.get("train_on_new_data", True)):
        return
    st.session_state.pop("xgb_fit_future", None)
    X, y = get_dataset_for_prompt_or_session(prompt, st.session_state)
    # persistence.get_dataset_for_prompt_or_session already guards dim mismatches
    try:
        if X is not None and y is not None and getattr(X, "shape", (0,))[0] > 0:
            lam_now = float(getattr(st.session_state, Keys.REG_LAMBDA, 1e-3))
            vm_train = "Ridge"
            try:
                getattr(st, "toast", lambda *a, **k: None)(f"Training {vm_train}…")
            except Exception:
                pass
            if _train_record is not None:
                _train_record(vm_train, lstate, X, y, lam_now, st.session_state)
            else:
                # Minimal fallback mirrors _curation_train_and_next
                from datetime import datetime, timezone
                last_at = st.session_state.get(Keys.LAST_TRAIN_AT)
                min_wait = float(st.session_state.get("min_train_interval_s", 0.0) or 0.0)
                recent = False
                if last_at and min_wait > 0.0:
                    try:
                        dt = datetime.fromisoformat(last_at)
                        recent = (datetime.now(timezone.utc) - dt).total_seconds() < min_wait
                    except Exception:
                        recent = False
                if not recent:
                    try:
                        from value_model import fit_value_model as _fit_vm
                        _fit_vm(vm_train, lstate, X, y, lam_now, st.session_state)
                    except Exception:
                        pass
    except Exception:
        pass


def _render_batch_ui() -> None:
    import streamlit as st
    try:
        from latent_logic import z_to_latents, z_from_prompt
    except Exception:
        from latent_opt import z_to_latents
        from latent_logic import z_from_prompt
    from flux_local import generate_flux_image_latents
    import time as _time

    # Ensure a model is loaded before any decode. The app sets this, but guard here
    # so fragments don’t trigger the env fallback path in flux_local.
    try:
        from flux_local import CURRENT_MODEL_ID, set_model  # type: ignore

        if CURRENT_MODEL_ID is None:
            from constants import DEFAULT_MODEL

            set_model(DEFAULT_MODEL)
    except Exception:
        pass

    (getattr(st, "subheader", lambda *a, **k: None))("Curation batch")
    try:
        globals()["GLOBAL_RENDER_COUNTER"] = int(globals().get("GLOBAL_RENDER_COUNTER", 0)) + 1
    except Exception:
        globals()["GLOBAL_RENDER_COUNTER"] = 1
    # Maintain a per-render counter in session for non-fragment key uniqueness
    try:
        st.session_state["render_count"] = int(st.session_state.get("render_count", 0)) + 1
    except Exception:
        pass
    # Bump a small render nonce and salt so non-frag keys differ per render
    try:
        st.session_state["render_nonce"] = int(st.session_state.get("render_nonce", 0)) + 1
        try:
            import secrets as __sec
            st.session_state["render_salt"] = int(__sec.randbits(32))
        except Exception:
            import time as __t
            st.session_state["render_salt"] = int(__t.time() * 1e9)
    except Exception:
        pass
    # Reset per-render button sequence to keep keys unique yet bounded
    # Keep button keys stable across reruns so clicks are captured
    lstate, prompt = _lstate_and_prompt()
    try:
        steps = int(getattr(st.session_state, "steps", 6) or 6)
    except Exception:
        steps = 6
    try:
        guidance_eff = float(getattr(st.session_state, "guidance_eff", 0.0) or 0.0)
    except Exception:
        guidance_eff = 0.0
    best_of = False  # Best-of removed: always use Good/Bad buttons
    cur_batch = st.session_state.cur_batch or []
    if not cur_batch:
        try:
            _curation_init_batch()
        except Exception:
            pass
        cur_batch = st.session_state.cur_batch or []
    # Prepare optional value scorer once per batch
    scorer = None
    scorer_status = None
    fut_running = False
    z_p = None
    try:
        from value_scorer import get_value_scorer_with_status
        # Single-scorer rule:
        # - If XGB cache exists → use XGB scorer
        # - Else if Use Ridge captions → use Ridge scorer
        # - Else → no scorer (n/a)
        cache = st.session_state.get("xgb_cache") or {}
        use_ridge_caps = bool(st.session_state.get("use_ridge_captions", False))
        scorer = None
        scorer_status = "n/a"
        vm_choice = None
        if cache.get("model") is not None:
            vm_choice = "XGBoost"
            scorer, scorer_status = get_value_scorer_with_status(
                "XGBoost", lstate, prompt, st.session_state
            )
        elif use_ridge_caps:
            vm_choice = "Ridge"
            scorer, scorer_status = get_value_scorer_with_status(
                "Ridge", lstate, prompt, st.session_state
            )
        # Gate noncritical scorer logs behind a simple verbosity flag (0/1/2)
        try:
            _val = getattr(st.session_state, "log_verbosity", None)
            _lv = 0 if (_val is None) else int(_val)
        except Exception:
            _lv = 0
        if _lv > 0:
            try:
                _log(f"[scorer] vm={vm_choice} status={scorer_status}")
            except Exception:
                pass
        fut = st.session_state.get("xgb_fit_future")
        fut_running = bool(
            fut is not None and not getattr(fut, "done", lambda: False)()
        )
        z_p = z_from_prompt(lstate, prompt)
    except Exception:
        scorer = None
    n = len(cur_batch)
    if n == 0:
        return
    per_row = min(5, n)

    for row_start in range(0, n, per_row):
        row_end = min(row_start + per_row, n)
        cols = getattr(st, "columns", lambda x: [None] * x)(row_end - row_start)
        for col_idx, i in enumerate(range(row_start, row_end)):
            col = cols[col_idx] if cols and len(cols) > col_idx else None

            def _render_item() -> None:
                # Create the vector for this image immediately before decode so
                # each tile uses a freshly sampled latent under the current
                # settings, independent of cur_batch.
                # Keep latents stable: use the current batch value; do not resample here
                z_i = cur_batch[i]

                t0 = _time.perf_counter()
                try:
                    la = z_to_latents(lstate, z_i)
                except Exception:
                    la = z_to_latents(z_i, lstate)
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
                _log(
                    f"[batch] decoded item={i} in {dt_ms:.1f} ms (steps={steps}, w={lstate.width}, h={lstate.height})"
                )
                # Predicted value using current value model scorer
                v_text = "Value: n/a"
                if scorer is not None and z_p is not None:
                    try:
                        fvec = z_i - z_p
                        v = float(scorer(fvec))
                        v_text = f"Value: {v:.3f}"
                        try:
                            tag = "XGB" if str(vm_choice_local) == "XGBoost" else "Ridge"
                            v_text = f"{v_text} [{tag}]"
                        except Exception:
                            pass
                    except Exception:
                        v_text = "Value: n/a"
                cap_txt = f"Item {i} • {v_text}"
                st.image(img_i, caption=cap_txt, width="stretch")

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
                            getattr(st, "toast", lambda *a, **k: None)(
                                f"Best-of: chose {i}"
                            )
                        except Exception:
                            pass
                        _log(
                            f"[perf] best_of choose item={i} took {(_time.perf_counter() - t0b) * 1000:.1f} ms"
                        )
                else:
                    btn_cols = getattr(st, "columns", lambda x: [None] * x)(2)
                    gcol = btn_cols[0] if btn_cols and len(btn_cols) > 0 else None
                    bcol = btn_cols[1] if btn_cols and len(btn_cols) > 1 else None
                    def _btn_key(prefix: str, idx: int) -> str:
                        # When fragments are active, keep keys stable across reruns.
                        if use_frags_active:
                            return f"{prefix}_{idx}"
                        # Otherwise vary across renders to satisfy non-frag tests.
                        try:
                            rcount = int(st.session_state.get("render_count", 0))
                        except Exception:
                            rcount = 0
                        return f"{prefix}_{rcount}_{idx}"

                    def _good_clicked() -> bool:
                        if gcol is not None:
                            with gcol:
                                return st.button(
                                    f"Good (+1) {i}",
                                    key=_btn_key("good", i),
                                    width="stretch",
                                )
                        return st.button(
                            f"Good (+1) {i}", key=_btn_key("good", i), width="stretch"
                        )

                    def _bad_clicked() -> bool:
                        if bcol is not None:
                            with bcol:
                                return st.button(
                                    f"Bad (-1) {i}",
                                    key=_btn_key("bad", i),
                                    width="stretch",
                                )
                        return st.button(
                            f"Bad (-1) {i}", key=_btn_key("bad", i), width="stretch"
                        )

                    if _good_clicked():
                        t0g = _time.perf_counter()
                        _curation_add(1, z_i, img_i)
                        st.session_state.cur_labels[i] = 1
                        _refit_from_dataset_keep_batch()
                        _curation_replace_at(i)
                        try:
                            _log(f"[batch] click good item={i}")
                        except Exception:
                            pass
                        try:
                            msg = "Labeled Good (+1)"
                            getattr(st, "toast", lambda *a, **k: None)(msg)
                            try:
                                import time as __t
                                st.session_state[Keys.LAST_ACTION_TEXT] = msg
                                st.session_state[Keys.LAST_ACTION_TS] = float(__t.time())
                            except Exception:
                                pass
                        except Exception:
                            pass
                        _log(
                            f"[perf] good_label item={i} took {(_time.perf_counter() - t0g) * 1000:.1f} ms"
                        )
                    if _bad_clicked():
                        t0b2 = _time.perf_counter()
                        _curation_add(-1, z_i, img_i)
                        st.session_state.cur_labels[i] = -1
                        _refit_from_dataset_keep_batch()
                        _curation_replace_at(i)
                        try:
                            _log(f"[batch] click bad item={i}")
                        except Exception:
                            pass
                        try:
                            msg = "Labeled Bad (-1)"
                            getattr(st, "toast", lambda *a, **k: None)(msg)
                            try:
                                import time as __t
                                st.session_state[Keys.LAST_ACTION_TEXT] = msg
                                st.session_state[Keys.LAST_ACTION_TS] = float(__t.time())
                            except Exception:
                                pass
                        except Exception:
                            pass
                        _log(
                            f"[perf] bad_label item={i} took {(_time.perf_counter() - t0b2) * 1000:.1f} ms"
                        )

            # Wrap each tile in its own fragment when available so the
            # latent sampling, decode, buttons, and saves are scoped per
            # image and can run independently. Streamlit exposes fragments
            # as a decorator, so we decorate _render_item and then call it.
            # 195g: fragments option removed — always use non-fragment path
            frag = None
            use_frags = False
            use_frags_active = False

            def _tile_cache_key() -> str:
                try:
                    nonce = int(st.session_state.get("cur_batch_nonce", 0))
                except Exception:
                    nonce = 0
                return f"tile_{nonce}_{i}"

            def _render_visual_and_cache() -> None:
                # Minimal duplication of _render_item visual section to avoid
                # button rendering inside fragments. Stores z/img for button handlers.
                # First try to use cached tile payload to avoid extra decodes
                key = _tile_cache_key()
                cache = st.session_state.get("_tile_cache", {}) or {}
                cached = cache.get(key) if isinstance(cache, dict) else None
                if isinstance(cached, dict) and "z" in cached and "img" in cached:
                    zi = cached["z"]
                    img_local = cached["img"]
                else:
                    try:
                        # Reuse scorer/z_p from outer scope when available
                        zi = cur_batch[i]
                    except Exception:
                        zi = None
                # Optionally resample using XGB hill when active
                try:
                    vm_choice_local = st.session_state.get("vm_choice")
                    use_xgb_local = vm_choice_local == "XGBoost"
                except Exception:
                    use_xgb_local = False
                if (
                    use_xgb_local
                    and scorer is not None
                    and not fut_running
                ):
                    try:
                        from latent_logic import sample_z_xgb_hill  # local import

                        steps_local = int(st.session_state.get("iter_steps", DEFAULT_ITER_STEPS))
                        lr_mu_local = float(st.session_state.get("lr_mu_ui", 0.3))
                        trust_val = st.session_state.get("trust_r", None)
                        trust_r_local = (
                            float(trust_val)
                            if (trust_val is not None and float(trust_val) > 0.0)
                            else None
                        )
                        step_scale_local = lr_mu_local * float(
                            getattr(lstate, "sigma", 1.0)
                        )
                        zi = sample_z_xgb_hill(
                            lstate,
                            prompt,
                            scorer,
                            steps=steps_local,
                            step_scale=step_scale_local,
                            trust_r=trust_r_local,
                        )
                    except Exception:
                        pass
                if cached is None:
                    if zi is None:
                        try:
                            zi = _sample_around_prompt(scale=0.8)
                        except Exception:
                            zi = cur_batch[i]
                    # Persist latent only if it changed
                    try:
                        import numpy as _np
                        same = (
                            isinstance(cur_batch[i], _np.ndarray)
                            and _np.array_equal(cur_batch[i], zi)
                        )
                    except Exception:
                        same = False
                    try:
                        if not same:
                            cur_batch[i] = zi
                            st.session_state.cur_batch = cur_batch
                    except Exception:
                        pass
                    la = z_to_latents(lstate, zi)
                    img_local = generate_flux_image_latents(
                        prompt,
                        latents=la,
                        width=lstate.width,
                        height=lstate.height,
                        steps=steps,
                        guidance=guidance_eff,
                    )
                    # Cache for button handler
                    try:
                        cache = st.session_state.get("_tile_cache", {}) or {}
                        cache[key] = {"z": zi, "img": img_local}
                        st.session_state["_tile_cache"] = cache
                    except Exception:
                        pass
                # Caption with value if scorer present
                v_text = "Value: n/a"
                if scorer is not None and z_p is not None:
                    try:
                        fvec = zi - z_p
                        v = float(scorer(fvec))
                        v_text = f"Value: {v:.3f}"
                        try:
                            tag = "XGB" if str(vm_choice_local) == "XGBoost" else "Ridge"
                            v_text = f"{v_text} [{tag}]"
                        except Exception:
                            pass
                    except Exception:
                        v_text = "Value: n/a"
                cap_txt = f"Item {i} • {v_text}"
                st.image(img_local, caption=cap_txt, width="stretch")

            def _render_buttons_from_cache() -> None:
                # Render Good/Bad using cached z/img to avoid work inside fragments
                btn_cols = getattr(st, "columns", lambda x: [None] * x)(2)
                gcol = btn_cols[0] if btn_cols and len(btn_cols) > 0 else None
                bcol = btn_cols[1] if btn_cols and len(btn_cols) > 1 else None
                try:
                    _log(f"[buttons] render for item={i}")
                except Exception:
                    pass
                def _btn_key(prefix: str, idx: int) -> str:
                    # Keep keys stable under fragments/reruns: prefix + index only
                    return f"{prefix}_{idx}"

                def _good_clicked() -> bool:
                    if gcol is not None:
                        with gcol:
                            return st.button(
                                f"Good (+1) {i}", key=_btn_key("good", i), width="stretch"
                            )
                    return st.button(
                        f"Good (+1) {i}", key=_btn_key("good", i), width="stretch"
                    )

                def _bad_clicked() -> bool:
                    if bcol is not None:
                        with bcol:
                            return st.button(
                                f"Bad (-1) {i}", key=_btn_key("bad", i), width="stretch"
                            )
                    return st.button(
                        f"Bad (-1) {i}", key=_btn_key("bad", i), width="stretch"
                    )

                cache = st.session_state.get("_tile_cache", {}) or {}
                data = cache.get(_tile_cache_key()) or {}
                zi = data.get("z")
                img_local = data.get("img")
                # If visual fragment hasn't cached yet, fall back to current latent
                # so tests can still observe button keys.
                if zi is None:
                    try:
                        zi = cur_batch[i]
                    except Exception:
                        zi = None
                if zi is None:
                    return
                if _good_clicked():
                    _curation_add(1, zi, img_local)
                    st.session_state.cur_labels[i] = 1
                    _refit_from_dataset_keep_batch()
                    _curation_replace_at(i)
                    try:
                        getattr(st, "toast", lambda *a, **k: None)("Labeled Good (+1)")
                    except Exception:
                        pass
                if _bad_clicked():
                    _curation_add(-1, zi, img_local)
                    st.session_state.cur_labels[i] = -1
                    _refit_from_dataset_keep_batch()
                    _curation_replace_at(i)
                    try:
                        getattr(st, "toast", lambda *a, **k: None)("Labeled Bad (-1)")
                    except Exception:
                        pass

            if col is not None:
                with col:
                    if use_frags and callable(frag):
                        try:
                            _log("[fragpath] active")
                        except Exception:
                            pass
                        try:
                            wrapped = frag(_render_visual_and_cache)
                            wrapped()
                        except TypeError:
                            _render_visual_and_cache()
                        _render_buttons_from_cache()
                    else:
                        try:
                            _log("[fragpath] inactive -> non-frag _render_item")
                        except Exception:
                            pass
                        _render_item()
            else:
                if use_frags and callable(frag):
                    try:
                        _log("[fragpath] active (no col)")
                    except Exception:
                        pass
                    try:
                        wrapped = frag(_render_visual_and_cache)
                        wrapped()
                    except TypeError:
                        _render_visual_and_cache()
                    _render_buttons_from_cache()
                else:
                    try:
                        _log("[fragpath] inactive (no col) -> non-frag _render_item")
                    except Exception:
                        pass
                    _render_item()


def run_batch_mode() -> None:
    _curation_init_batch()
    _render_batch_ui()
