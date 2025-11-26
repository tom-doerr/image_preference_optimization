from __future__ import annotations

from typing import Any, Tuple
import time as _time
import numpy as np
from ipo.infra.constants import Keys
import logging as _logging
from .batch_util import save_and_print as _save_and_print

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
    """Minimal gated log: respects LOG_VERBOSITY env (0/1/2)."""
    import os as _os
    lv = int((_os.getenv("LOG_VERBOSITY") or "0") or "0")
    if lv <= 0:
        return
    print(msg)
    getattr(LOGGER, level, LOGGER.info)(msg)


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
        from ipo.infra.constants import DEFAULT_PROMPT

        prompt = DEFAULT_PROMPT
    return lstate, prompt


def _vm_and_cache(st):
    try:
        vm_choice = str(st.session_state.get(Keys.VM_CHOICE, ""))
    except Exception:
        vm_choice = ""
    # XGB session cache removed; ignore any legacy keys
    return vm_choice, {}


def _ridge_norm(lstate) -> float:
    try:
        import numpy as _np

        w = getattr(lstate, "w", None)
        return float(_np.linalg.norm(w)) if w is not None else 0.0
    except Exception:
        return 0.0


def _compute_zp(lstate, prompt):
    try:
        from latent_logic import z_from_prompt as _zfp

        return _zfp(lstate, prompt)
    except Exception:
        return None


def _try_distance(_gvs, vm_choice, lstate, prompt, st):
    if vm_choice != "Distance":
        return None, None
    s, tag = _gvs("Distance", lstate, prompt, st.session_state)
    return (s, tag if s is not None else None)


def _try_logistic(_gvs, vm_choice, lstate, prompt, st):
    if vm_choice != "Logistic":
        return None, None
    s, tag = _gvs("Logistic", lstate, prompt, st.session_state)
    return (s, tag) if s is not None else (None, None)


def _try_xgb_live(_gvs, lstate, prompt, st):
    # No explicit cache: ask the scorer builder directly; returns None when unavailable
    s, tag = _gvs("XGBoost", lstate, prompt, st.session_state)
    return (s, tag) if s is not None else (None, None)


def _try_ridge_if_norm(_gvs, lstate, prompt, st):
    if _ridge_norm(lstate) <= 0.0:
        return None, None
    s, tag = _gvs("Ridge", lstate, prompt, st.session_state)
    return (s, tag) if (s is not None and tag == "Ridge") else (None, None)


def _pick_scorer(vm_choice: str, cache, lstate, prompt, st):
    """Internal: decide scorer without side effects.
    Rule (simplified):
    - XGBoost → use XGB only; no Ridge/Logit fallback for captions.
    - Logistic/Distance → use their respective scorer.
    - Ridge → use Ridge when ‖w‖>0.
    """
    try:
        from value_scorer import get_value_scorer as _gvs
    except Exception:
        return None, None
    if str(vm_choice) == "XGBoost":
        s, tag = _try_xgb_live(_gvs, lstate, prompt, st)
        return (s, tag) if s is not None else (None, None)
    # Distance / Logistic first
    s, tag = _try_distance(_gvs, vm_choice, lstate, prompt, st)
    if s is not None:
        return s, tag
    s, tag = _try_logistic(_gvs, vm_choice, lstate, prompt, st)
    if s is not None:
        return s, tag
    # Ridge only when explicitly not XGB and w is non‑zero
    s, tag = _try_ridge_if_norm(_gvs, lstate, prompt, st)
    if s is not None:
        return s, tag
    return None, None


def _choose_scorer(st, lstate, prompt):
    """Select a value scorer and tag based on session state (no auto-fit).

    Returns (scorer_callable_or_None, tag_or_None, z_prompt_vector_or_None).
    """
    vm_choice, cache = _vm_and_cache(st)
    scorer, scorer_tag = _pick_scorer(vm_choice, cache, lstate, prompt, st)
    z_p = _compute_zp(lstate, prompt)
    return scorer, scorer_tag, z_p


def _predict_value(scorer, z_p, z_i):
    try:
        if scorer is None or z_p is None:
            return None
        fvec = z_i - z_p
        return float(scorer(fvec))
    except Exception:
        return None


def _vm_tag(st) -> str:
    try:
        vmn = str(getattr(st.session_state, "vm_choice", ""))
        return (
            "Distance" if vmn == "Distance" else
            "XGB" if vmn == "XGBoost" else
            "Logit" if vmn == "Logistic" else
            "Ridge"
        )
    except Exception:
        return "Ridge"


def _tile_value_text(st, z_p, z_i, scorer) -> str:
    """Return "Value: … [TAG]" from the active scorer; never fall back when VM=XGBoost."""
    v = _predict_value(scorer, z_p, z_i)
    if v is not None:
        return f"Value: {v:.3f} [{_vm_tag(st)}]"
    # If user selected Logistic explicitly, allow Logit fallback
    if str(getattr(st.session_state, "vm_choice", "")) == "Logistic":
        alt = _maybe_logit_value(z_p, z_i, st)
        if alt:
            return alt
    return "Value: n/a"


def _render_tiles_row(
    st,
    idxs,
    lstate,
    prompt: str,
    steps: int,
    guidance_eff: float,
    best_of: bool,
    scorer,
    cur_batch,
    z_p,
):
    try:
        rn = int(st.session_state.get("render_nonce", 0))
    except Exception:
        rn = 0
    cols = getattr(st, "columns", lambda x: [None] * x)(len(idxs))
    for col, i in zip(cols, idxs):
        if col is not None:
            with col:
                _render_batch_tile_body(
                    int(i),
                    rn,
                    lstate,
                    prompt,
                    int(steps),
                    float(guidance_eff),
                    bool(best_of),
                    scorer,
                    False,
                    cur_batch,
                    z_p,
                )
        else:
            _render_batch_tile_body(
                int(i),
                rn,
                lstate,
                prompt,
                int(steps),
                float(guidance_eff),
                bool(best_of),
                scorer,
                False,
                cur_batch,
                z_p,
            )


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
    """Return (scorer, status) for XGB from cache (no auto-fit)."""
    # Prefer unified scorer; provide a tiny compat shim when tests stub only the old API.
    try:
        from value_scorer import get_value_scorer as _gvs
    except Exception:
        from value_scorer import get_value_scorer_with_status as _gvs_ws  # type: ignore

        def _gvs(vm_choice, lstate, prompt, session_state):  # type: ignore
            s, status = _gvs_ws(vm_choice, lstate, prompt, session_state)
            return (s, ("ok" if callable(s) else status))

    try:
        scorer, tag_or_status = _gvs(
            "XGBoost", lstate, prompt, __import__("streamlit").session_state
        )
        return (scorer, "ok") if scorer is not None else (None, str(tag_or_status))
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
    """Thin wrapper around batch_util.read_curation_params."""
    import streamlit as st
    try:
        from .batch_util import read_curation_params as _rcp
        return _rcp(st, Keys, default_steps=10, default_lr_mu=0.3)
    except Exception:
        # Deterministic fallback
        return (str(st.session_state.get(Keys.VM_CHOICE) or ""), False, 10, 0.3, None)


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
    st.session_state["cur_batch_nonce"] = int(st.session_state.get("cur_batch_nonce", 0)) + 1
    dt_ms = (_time.perf_counter() - t0) * 1000.0
    vm_choice = st.session_state.get(Keys.VM_CHOICE)
    _log(
        f"[batch] new batch: n={len(z_list)} d={lstate.d} sigma={lstate.sigma:.3f} ‖z_p‖={float(np.linalg.norm(z_p)):.3f} size={lstate.width}x{lstate.height} vm={vm_choice} in {dt_ms:.1f} ms"
    )


def _curation_replace_at(idx: int) -> None:
    """Deterministically resample the item at idx; keep batch size constant.

    Under full app runs we can refresh the entire batch; in stubs/tests we
    resample only the requested index to keep behavior predictable.
    """
    import streamlit as st

    zs = list(getattr(st.session_state, "cur_batch", []) or [])
    if not zs:
        _curation_new_batch()
        zs = list(getattr(st.session_state, "cur_batch", []) or [])
        if not zs:
            return
    i = int(idx) % len(zs)
    _log(f"[batch] replace_at idx={i} nonce={int(st.session_state.get('cur_batch_nonce', 0))}")
    print(f"[batch] replace_at idx={i}")
    # Deterministic resample keyed on (batch_nonce, idx)
    zi = _resample_tile_at_index(i)
    from .batch_util import set_batch_item as _set_batch_item
    _set_batch_item(st, i, zi)


def _append_mem_dataset(st, Keys, feat: np.ndarray, label: float) -> None:
    try:
        X = st.session_state.get("dataset_X")
        y = st.session_state.get("dataset_y")
        lab = np.array([float(label)])
        st.session_state.dataset_X = feat if X is None else np.vstack([X, feat])
        st.session_state.dataset_y = lab if y is None else np.concatenate([y, lab])
        st.session_state[Keys.DATASET_X] = st.session_state.dataset_X
        st.session_state[Keys.DATASET_Y] = st.session_state.dataset_y
    except Exception:
        pass


def _record_last_action_and_step(st, Keys, lstate, msg: str) -> None:
    try:
        import time as _time
        st.session_state[Keys.LAST_ACTION_TEXT] = msg
        st.session_state[Keys.LAST_ACTION_TS] = float(_time.time())
    except Exception:
        pass
    try:
        setattr(lstate, "step", int(getattr(lstate, "step", 0)) + 1)
    except Exception:
        pass


def _update_rows_display(st, Keys) -> None:
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


def _resample_tile_at_index(i: int) -> np.ndarray:
    """Return a deterministic resample for tile i based on batch nonce and prompt.

    Falls back to around-prompt sampling when latent logic is unavailable.
    """
    import streamlit as st
    import numpy as _np
    try:
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
        return z_p + float(getattr(lstate, 'sigma', 1.0)) * 0.8 * r
    except Exception:
        return _sample_around_prompt(scale=0.8)


def _cooldown_recent(st) -> bool:
    try:
        from .batch_util import cooldown_recent as _cd
        return _cd(st, Keys)
    except Exception:
        return False


def _fit_ridge_once(lstate, X, y, lam_now, st) -> None:
    try:
        from ipo.core.value_model import fit_value_model as _fit_vm
        _fit_vm("Ridge", lstate, X, y, float(lam_now), st.session_state)
    except Exception:
        pass


def _curation_add(label: int, z: np.ndarray, img=None) -> None:
    import streamlit as st
    from ipo.infra.constants import Keys
    from latent_logic import z_from_prompt

    lstate, prompt = _lstate_and_prompt()
    z_p = z_from_prompt(lstate, prompt)
    feat = (z - z_p).reshape(1, -1)
    try:
        _log(f"[data] append label={int(label)} ‖feat‖={float(np.linalg.norm(feat)):.3f}")
    except Exception:
        pass
    # Unconditional short debug for tests capturing stdout
    try:
        print(f"[data] append label={int(label)}")
    except Exception:
        pass
    _append_mem_dataset(st, Keys, feat, float(label))
    try:
        _row_idx, _save_dir, msg = _save_and_print(prompt, feat, float(label), img, st)
        _record_last_action_and_step(st, Keys, lstate, msg)
        _update_rows_display(st, Keys)
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
    from .batch_tiles import render_batch_tile_body as _rb
    _rb(i, render_nonce, lstate, prompt, steps, guidance_eff, best_of, scorer, fut_running, cur_batch, z_p)


def _curation_train_and_next() -> None:
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    if not bool(st.session_state.get("train_on_new_data", True)):
        _curation_new_batch()
        return
    # Resolve dataset (memory-first) and maybe train once
    from ipo.ui.ui_sidebar import _get_dataset_for_display as _gdf
    X, y = _gdf(st, lstate, prompt)
    _maybe_train_ridge_sync(st, lstate, X, y)
    _curation_new_batch()


def _refit_from_dataset_keep_batch() -> None:
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    if not bool(st.session_state.get("train_on_new_data", True)):
        return
    from ipo.ui.ui_sidebar import _get_dataset_for_display as _gdf
    X, y = _gdf(st, lstate, prompt)
    _maybe_train_ridge_sync(st, lstate, X, y)

def _maybe_train_ridge_sync(st, lstate, X, y) -> None:
    """Train Ridge synchronously once when data are present and cooldown allows."""
    try:
        n = int(getattr(X, "shape", (0,))[0]) if X is not None else 0
        if n <= 0 or y is None:
            return
        lam_now = float(getattr(st.session_state, Keys.REG_LAMBDA, 1e300))
        getattr(st, "toast", lambda *a, **k: None)("Training Ridge…")
        if not _cooldown_recent(st):
            _fit_ridge_once(lstate, X, y, lam_now, st)
    except Exception:
        pass


def _render_batch_ui() -> None:
    import streamlit as st

    # Header + init
    lstate, prompt, steps, guidance_eff, cur_batch, scorer, z_p = _batch_init(st)
    best_of = False  # Best-of removed: always use Good/Bad buttons
    n = len(cur_batch)
    if n == 0:
        return
    per_row = min(5, n)

    for row_start in range(0, n, per_row):
        row_end = min(row_start + per_row, n)
        # Use unified tile renderer for the row
        try:
            _render_tiles_row(
                st,
                list(range(row_start, row_end)),
                lstate,
                prompt,
                steps,
                guidance_eff,
                best_of,
                scorer,
                cur_batch,
                z_p,
            )
            continue
        except Exception:
            pass
        # Fallback: inline render (unchanged behavior) via helper
        _render_row_fallback(
            st,
            row_start,
            row_end,
            lstate,
            prompt,
            steps,
            guidance_eff,
            best_of,
            scorer,
            cur_batch,
            z_p,
        )

            

            # (Fragments/caching path removed in simplification.)


def run_batch_mode() -> None:
    _curation_init_batch()
    _render_batch_ui()
def _decode_one(i: int, lstate: Any, prompt: str, z_i: np.ndarray, steps: int, guidance_eff: float):
    from .batch_decode import decode_one as _dec
    return _dec(i, lstate, prompt, z_i, steps, guidance_eff)


def _render_row_fallback(
    st,
    row_start: int,
    row_end: int,
    lstate,
    prompt: str,
    steps: int,
    guidance_eff: float,
    best_of: bool,
    scorer,
    cur_batch,
    z_p,
) -> None:
    import time as _time
    cols = getattr(st, "columns", lambda x: [None] * x)(row_end - row_start)
    for col_idx, i in enumerate(range(row_start, row_end)):
        col = cols[col_idx] if cols and len(cols) > col_idx else None

        def _render_item() -> None:
            z_i = cur_batch[i]
            img_i = _decode_one(i, lstate, prompt, z_i, steps, guidance_eff)
            # Predicted value using current value model scorer
            v_text = "Value: n/a"
            try:
                if scorer is not None and z_p is not None:
                    fvec = z_i - z_p
                    v = float(scorer(fvec))
                    v_text = f"Value: {v:.3f}"
                    # Use existing VM tag helper for consistency
                    v_text = f"{v_text} [{_vm_tag(st)}]"
            except Exception:
                v_text = "Value: n/a"
            if v_text == "Value: n/a":
                v_text = _maybe_logit_value(z_p, z_i, st) or v_text
            cap_txt = f"Item {i} • {v_text}"
            st.image(img_i, caption=cap_txt, width="stretch")

            btn_cols = getattr(st, "columns", lambda x: [None] * x)(2)
            gcol = btn_cols[0] if btn_cols and len(btn_cols) > 0 else None
            bcol = btn_cols[1] if btn_cols and len(btn_cols) > 1 else None

            def _btn_key(prefix: str, idx: int) -> str:
                try:
                    rcount = int(st.session_state.get("render_count", 0))
                except Exception:
                    rcount = 0
                return f"{prefix}_{rcount}_{idx}"

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

            if _good_clicked():
                t0g = _time.perf_counter()
                _curation_add(1, z_i, img_i)
                st.session_state.cur_labels[i] = 1
                _refit_from_dataset_keep_batch()
                _curation_replace_at(i)
                try:
                    msg = "Labeled Good (+1)"
                    getattr(st, "toast", lambda *a, **k: None)(msg)
                except Exception:
                    pass
                try:
                    from ipo.infra.constants import Keys
                    import time as __t
                    st.session_state[Keys.LAST_ACTION_TEXT] = msg
                    st.session_state[Keys.LAST_ACTION_TS] = float(__t.time())
                except Exception:
                    pass
                _log(
                    f"[perf] good_label item={i} took {(_time.perf_counter() - t0g) * 1000:.1f} ms"
                )
            if _bad_clicked():
                _label_and_replace(i, -1, z_i, img_i, st)

        # Always use non-fragment path
        if col is not None:
            with col:
                _render_item()
        else:
            _render_item()

def _maybe_logit_value(z_p, z_i, st) -> str | None:
    try:
        from ipo.infra.constants import Keys as _K
        w = st.session_state.get(_K.LOGIT_W)
        if w is not None and z_p is not None:
            import numpy as _np
            wv = _np.asarray(w, dtype=float)
            fvec = z_i - z_p
            zlog = float(_np.dot(wv, fvec))
            p = float(1.0 / (1.0 + _np.exp(-zlog)))
            return f"Value: {p:.3f} [Logit]"
    except Exception:
        return None
    return None


def _label_and_replace(i: int, lbl: int, z_i: np.ndarray, img_i, st) -> None:
    from .batch_buttons import _label_and_replace as _lr
    _lr(i, lbl, z_i, img_i, st)


def _render_good_bad_buttons(st, i: int, z_i: np.ndarray, img_i, nonce: int, gcol, bcol) -> None:
    from .batch_buttons import render_good_bad_buttons as _rgb
    _rgb(st, i, z_i, img_i, nonce, gcol, bcol)

def _ensure_model_ready() -> None:
    from .batch_util import ensure_model_ready as _emr
    _emr()


def _prep_render_counters(st) -> None:
    from .batch_util import prep_render_counters as _prc
    _prc(st)

def _button_key(st, prefix: str, nonce: int, idx: int) -> str:
    from .batch_buttons import _button_key as _bk
    return _bk(st, prefix, nonce, idx)


def _toast_and_record(st, msg: str) -> None:
    from .batch_buttons import _toast_and_record as _tr
    _tr(st, msg)


def _handle_best_of(st, i: int, img_i, cur_batch) -> None:
    from .batch_buttons import handle_best_of as _hb
    _hb(st, i, img_i, cur_batch)


def _batch_init(st):
    """Header + counters + state init for the batch UI.

    Returns (lstate, prompt, steps, guidance_eff, cur_batch, scorer, z_p).
    """
    # Ensure a model is loaded before any decode.
    _ensure_model_ready()
    (getattr(st, "subheader", lambda *a, **k: None))("Curation batch")
    _prep_render_counters(st)
    lstate, prompt = _lstate_and_prompt()
    steps = int(getattr(st.session_state, "steps", 6) or 6)
    guidance_eff = float(getattr(st.session_state, "guidance_eff", 0.0) or 0.0)
    cur_batch = getattr(st.session_state, "cur_batch", []) or []
    if not cur_batch:
        _curation_init_batch()
        cur_batch = getattr(st.session_state, "cur_batch", []) or []
    scorer, _scorer_tag, z_p = _choose_scorer(st, lstate, prompt)
    return lstate, prompt, steps, guidance_eff, cur_batch, scorer, z_p
