import streamlit as st
import logging as _logging
try:
    from rich_cli import enable_color_print as _enable_color
    _enable_color()
except Exception:
    pass
import numpy as np
from typing import Optional, Tuple
import os
import hashlib
from PIL import Image
from constants import (
    DEFAULT_PROMPT,
    DEFAULT_MODEL,
    MODEL_CHOICES,
)
from constants import Config, Keys
from ui import sidebar_metric_rows
import batch_ui as _batch_ui
from ui_metrics import render_iter_step_scores, render_mu_value_history
from ui_controls import build_batch_controls, build_queue_controls, build_size_controls, build_pair_controls
from persistence import state_path_for_prompt, export_state_bytes, dataset_rows_for_prompt, dataset_stats_for_prompt
import background as bg
from persistence_ui import render_persistence_controls
from latent_opt import (
    init_latent_state,
    propose_next_pair,
    z_from_prompt,
    update_latent_ridge,
    save_state,
    load_state,
    propose_latent_pair_ridge,
)
from value_model import fit_value_model
import latent_logic as ll  # module alias for patchable ridge_fit in tests
from flux_local import (
    generate_flux_image_latents,
    set_model,
)
from pair_ui import (
    generate_pair as _pair_generate,
    _prefetch_next_for_generate as _pair_prefetch,
    _pair_scores as _pair_scores_impl,
)
from modes import run_mode
# Optional helpers (text-only path and debug accessor) — may be absent in tests
try:
    from flux_local import generate_flux_image  # type: ignore
except Exception:  # pragma: no cover - shim for minimal stubs
    generate_flux_image = None  # type: ignore
try:
    from flux_local import get_last_call  # type: ignore
except Exception:  # pragma: no cover
    def get_last_call():  # type: ignore
        return {}

st.set_page_config(page_title="Latent Preference Optimizer", layout="wide")
# Streamlit rerun API shim: prefer st.rerun(), fallback to experimental in older versions
st_rerun = getattr(st, 'rerun', getattr(st, 'experimental_rerun', None))

def _toast(msg: str) -> None:
    """Lightweight toast; falls back to sidebar write in test stubs."""
    t = getattr(st, 'toast', None)
    if callable(t):
        try:
            t(str(msg))
            return
        except Exception:
            pass
    # fallback for stubs
    try:
        st.sidebar.write(str(msg))
    except Exception:
        pass

# Shared logger routed to ipo.debug.log (and stdout for tests)
LOGGER = _logging.getLogger("ipo")
if not LOGGER.handlers:
    try:
        _h = _logging.FileHandler("ipo.debug.log")
        _h.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s app: %(message)s"))
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

def image_to_z(img: Image.Image, lstate) -> np.ndarray:
    """Convert an uploaded RGB image to a latent vector matching lstate.d."""
    h8, w8 = lstate.height // 8, lstate.width // 8
    arr = np.asarray(img.convert("RGB").resize((w8, h8)))
    arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
    pad = np.zeros((h8, w8, 1), dtype=np.float32)
    arr = np.concatenate([arr, pad], axis=2)
    arr = arr - arr.mean(axis=(0, 1), keepdims=True)
    lat = arr.transpose(2, 0, 1)  # (4, h8, w8)
    return lat.reshape(-1)

# Back-compat for tests: keep names on app module
_state_path_for_prompt = state_path_for_prompt

# Prompt-aware persistence
if 'prompt' not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT
# Default: train value models in background to keep UI responsive.
if 'xgb_train_async' not in st.session_state:
    st.session_state.xgb_train_async = True
# Also default Ridge to async to avoid UI stalls during fits.
if Keys.RIDGE_TRAIN_ASYNC not in st.session_state:  # keep minimal logic
    st.session_state[Keys.RIDGE_TRAIN_ASYNC] = True

_sb_txt = getattr(st.sidebar, "text_input", st.text_input)
base_prompt = _sb_txt("Prompt", value=st.session_state.prompt)
prompt_changed = (base_prompt != st.session_state.prompt)
if prompt_changed:
    st.session_state.prompt = base_prompt

st.session_state.state_path = state_path_for_prompt(st.session_state.prompt)


# Fragmented image renderer with a no-op fallback when st.fragment is unavailable (e.g., in tests)
def _image_fragment(img, caption: str, v_label: str | None = None, v_val: float | None = None) -> None:
    """Render a single image (and optional metric) in its own fragment when supported."""
    if img is None:
        st.empty()
        return
    st.image(img, caption=caption, width="stretch")
    if v_val is not None:
        try:
            st.metric(v_label or "V", f"{v_val:.3f}")
        except Exception:
            st.write(f"{v_label or 'V'}: {v_val:.3f}")

# Apply fragment decorator dynamically if available
_frag = getattr(st, 'fragment', None)
try:
    _use_frags = bool(getattr(st.session_state, 'use_fragments', True))
except Exception:
    _use_frags = True
if _use_frags and callable(_frag):
    _image_fragment = _frag(_image_fragment)

def _init_pair_for_state(new_state) -> None:
    try:
        # Proposer is tied to the Value model to simplify UI:
        # CosineHill when selected, otherwise DistanceHill.
        vmc = st.session_state.get('vm_choice', 'DistanceHill')
        pp = 'CosineHill' if vmc == 'CosineHill' else 'DistanceHill'
        if pp == 'DistanceHill':
            from persistence import get_dataset_for_prompt_or_session
            Xd, yd = get_dataset_for_prompt_or_session(st.session_state.prompt, st.session_state)
            from latent_logic import propose_pair_distancehill
            z1, z2 = propose_pair_distancehill(new_state, st.session_state.prompt, Xd, yd, alpha=0.5, gamma=0.5, trust_r=None)
        elif pp == 'CosineHill':
            from persistence import get_dataset_for_prompt_or_session
            Xd, yd = get_dataset_for_prompt_or_session(st.session_state.prompt, st.session_state)
            from latent_logic import propose_pair_cosinehill
            z1, z2 = propose_pair_cosinehill(new_state, st.session_state.prompt, Xd, yd, alpha=0.5, beta=5.0, trust_r=None)
        else:
            z1, z2 = propose_next_pair(new_state, st.session_state.prompt, opts=_proposer_opts())
        st.session_state.lz_pair = (z1, z2)
        # Log proposer used
        try:
            from datetime import datetime, timezone
            log = st.session_state.get('pair_log') or []
            log.append({'when': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'proposer': str(pp)})
            st.session_state.pair_log = log
        except Exception:
            pass
    except Exception:
        st.session_state.lz_pair = propose_latent_pair_ridge(new_state)


def _reset_derived_state(new_state) -> None:
    st.session_state.images = (None, None)
    st.session_state.mu_image = None
    if getattr(new_state, 'mu_hist', None) is not None and new_state.mu_hist.size > 0:
        st.session_state.mu_history = [m.copy() for m in new_state.mu_hist]
    else:
        st.session_state.mu_history = [new_state.mu.copy()]
    st.session_state.mu_best_idx = 0
    st.session_state.prompt_image = None
    st.session_state.pop('next_prefetch', None)
    st.session_state.pop('_bg_exec', None)
    try:
        bg.reset_executor()
    except Exception:
        pass


def _apply_state(new_state) -> None:
    """Apply a freshly loaded/created state to session and reset derived caches."""
    st.session_state.lstate = new_state
    try:
        use_rand = bool(getattr(st.session_state, "use_random_anchor", False))
        setattr(new_state, "use_random_anchor", use_rand)
        setattr(new_state, "random_anchor_z", None)
    except Exception:
        pass
    _init_pair_for_state(new_state)
    _reset_derived_state(new_state)
    st.session_state.images = (None, None)
    st.session_state.mu_image = None
    if getattr(new_state, 'mu_hist', None) is not None and new_state.mu_hist.size > 0:
        st.session_state.mu_history = [m.copy() for m in new_state.mu_hist]
    else:
        st.session_state.mu_history = [new_state.mu.copy()]
    st.session_state.mu_best_idx = 0
    # Generate prompt-only image now only when text path is unavailable
    # (at runtime the autorun block below will prefer the text path, which is more reliable)
    # Reset prompt image; autorun below will generate via text path
    st.session_state.prompt_image = None
    # Clear any pending prefetch when state changes
    st.session_state.pop('next_prefetch', None)
    # Reset shared background executor
    st.session_state.pop('_bg_exec', None)
    try:
        bg.reset_executor()
    except Exception:
        pass

if 'lstate' not in st.session_state or prompt_changed:
    if os.path.exists(st.session_state.state_path):
        _apply_state(load_state(st.session_state.state_path))
    else:
        _apply_state(init_latent_state())
    if 'prompt_image' not in st.session_state:
        st.session_state.prompt_image = None

lstate = st.session_state.lstate
z_a, z_b = st.session_state.lz_pair
_sb_num = getattr(st.sidebar, 'number_input', st.number_input)
_sb_sld = getattr(st.sidebar, 'slider', st.slider)

# Top-of-sidebar: Generation mode and Value model
_sb_sel = getattr(st.sidebar, 'selectbox', None)
st.sidebar.subheader("Mode & value model")
_gen_opts = ["Batch curation", "Async queue", "Upload latents"]
selected_gen_mode = None
if callable(_sb_sel):
    try:
        selected_gen_mode = _sb_sel("Generation mode", _gen_opts, index=0)
        if selected_gen_mode not in _gen_opts:
            selected_gen_mode = None
    except Exception:
        selected_gen_mode = None

_vm_opts = ["XGBoost", "DistanceHill", "Ridge", "CosineHill"]
vm_choice = None
if callable(_sb_sel):
    try:
        vm_choice = _sb_sel("Value model", _vm_opts, index=0)
        if vm_choice not in _vm_opts:
            vm_choice = "XGBoost"
    except Exception:
        vm_choice = "XGBoost"
else:
    vm_choice = "XGBoost"
st.session_state[Keys.VM_CHOICE] = vm_choice
# Training uses the active value model choice to stay in sync.
st.session_state[Keys.VM_TRAIN_CHOICE] = vm_choice

# Batch/queue controls near top for quick access
_exp = getattr(st.sidebar, 'expander', None)
batch_size = None
queue_size = None
if selected_gen_mode == _gen_opts[0]:  # Batch curation
    batch_size = build_batch_controls(st, expanded=True)
elif selected_gen_mode == _gen_opts[1]:  # Async queue
    queue_size = build_queue_controls(st, expanded=True)
else:
    batch_size = build_batch_controls(st, expanded=False)
    queue_size = build_queue_controls(st, expanded=False)
try:
    if queue_size is not None:
        st.session_state[Keys.QUEUE_SIZE] = int(queue_size)
except Exception:
    pass
try:
    st.session_state[Keys.BATCH_SIZE] = int(batch_size)
except Exception:
    pass

# Optional: random latent anchor instead of prompt-derived anchor
try:
    _rand_anchor_default = bool(getattr(st.session_state, "use_random_anchor", True))
except Exception:
    _rand_anchor_default = True
try:
    _rand_anchor_cb = getattr(st.sidebar, "checkbox", lambda *a, **k: _rand_anchor_default)(
        "Use random anchor (ignore prompt)", value=_rand_anchor_default
    )
except Exception:
    _rand_anchor_cb = _rand_anchor_default
try:
    use_random_anchor = bool(_rand_anchor_cb)
    st.session_state["use_random_anchor"] = use_random_anchor
    setattr(lstate, "use_random_anchor", use_random_anchor)
    # Reset cached random anchor when toggled on so a fresh one is drawn.
    if use_random_anchor:
        try:
            delattr(lstate, "random_anchor_z")
        except Exception:
            setattr(lstate, "random_anchor_z", None)
except Exception:
    pass

# Pair proposer dropdown removed; proposer is derived from Value model.

"""Top-of-sidebar: Training data & scores (rows metric auto-refreshes in a fragment)."""
try:
    st.sidebar.subheader("Training data & scores")

    def _rows_metric_only() -> None:
        from ui import sidebar_metric
        # Prefer live session rows; fall back to disk
        try:
            rows_live = int(len(st.session_state.get("dataset_y", []) or []))
        except Exception:
            rows_live = 0
        try:
            rows_disk = int(dataset_rows_for_prompt(base_prompt))
        except Exception:
            rows_disk = 0
        n_rows = max(rows_live, rows_disk)
        # Small spinner to show liveness
        try:
            import time as _time
            _spin = "|/-\\"
            _art = _spin[int(_time.time()) % len(_spin)]
            disp = f"{n_rows} {_art}"
        except Exception:
            disp = str(n_rows)
        sidebar_metric("Dataset rows", disp)
        # Scope refresh to this fragment when available
        try:
            _ar = getattr(st, 'autorefresh', None)
            if callable(_ar):
                _ar(interval=1000, key='rows_auto_refresh')
        except Exception:
            pass

    # Wrap rows metric in a fragment when available
    _frag = getattr(st, 'fragment', None)
    use_frags = bool(getattr(st.session_state, 'use_fragments', True))
    if use_frags and callable(_frag):
        try:
            _frag(_rows_metric_only)()
        except TypeError:
            _rows_metric_only()
    else:
        _rows_metric_only()
    # Train score using selected value model (Ridge/XGBoost)
    try:
        # Prefer on-disk dataset; fall back to in-memory X/y if present, but
        # refuse to use rows whose feature dim does not match the current latent dim.
        from persistence import get_dataset_for_prompt_or_session as _get_ds
        X_, y_ = _get_ds(base_prompt, st.session_state)
        if X_ is not None and y_ is not None and getattr(X_, "shape", (0, 0))[0] > 0:
            try:
                d_x = int(getattr(X_, "shape", (0, 0))[1])
                d_lat = int(getattr(lstate, "d", d_x))
                if d_x != d_lat:
                    st.session_state["dataset_dim_mismatch"] = (d_x, d_lat)
                    X_, y_ = None, None
            except Exception:
                X_, y_ = None, None
        # Lazy auto-fit: if we have a usable dataset but no value model yet,
        # delegate to value_model.ensure_fitted so there is a single place
        # that decides when Ridge/XGB trains.
        if X_ is not None and y_ is not None and getattr(X_, "shape", (0,))[0] > 0:
            try:
                from value_model import ensure_fitted as _ensure_fitted
                lam_auto = float(st.session_state.get("reg_lambda", 1e-3))
                _ensure_fitted(vm_choice, lstate, X_, y_, lam_auto, st.session_state)
            except Exception:
                pass
        if (X_ is None or getattr(X_, "size", 0) == 0) and (y_ is None or getattr(y_, "size", 0) == 0):
            X_ = getattr(lstate, 'X', None)
            y_ = getattr(lstate, 'y', None)
        _train_score = "n/a"
        if X_ is not None and y_ is not None and getattr(X_, "shape", (0,))[0] > 0:
            # Prefer XGB when selected and a model is cached
            _use_xgb_now = (vm_choice == "XGBoost")
            try:
                cache = st.session_state.get('xgb_cache') or {}
                mdl = cache.get('model')
            except Exception:
                mdl = None
            if _use_xgb_now and mdl is not None:
                try:
                    from xgb_value import score_xgb_proba  # type: ignore
                    import numpy as _np
                    probs = _np.array([score_xgb_proba(mdl, fv) for fv in X_], dtype=float)
                    preds = probs >= 0.5
                    _acc = float(_np.mean(preds == (y_ > 0)))
                    _train_score = f"{_acc*100:.0f}%"
                except Exception:
                    pass
            if _train_score == "n/a":
                # Ridge fallback
                _pred = X_ @ lstate.w
                _acc = float(( (_pred >= 0) == (y_ > 0)).mean())
                _train_score = f"{_acc*100:.0f}%"
    except Exception:
        _train_score = "n/a"
    # Last train time (ISO) if available
    try:
        _last_train = str(st.session_state.get('last_train_at')) if st.session_state.get('last_train_at') else 'n/a'
    except Exception:
        _last_train = 'n/a'
    # Value scorer status summary (which scorer + status + rows)
    try:
        from value_scorer import get_value_scorer_with_status as _gss
        _, _vs_status = _gss(vm_choice, lstate, base_prompt, st.session_state)
        _vs_name = str(vm_choice or "Ridge")
        _vs_rows = 0
        if X_ is not None and y_ is not None and getattr(X_, "shape", (0,))[0] > 0:
            _vs_rows = int(getattr(X_, "shape", (0,))[0])
        _vs_line = f"{_vs_name} ({_vs_status}, rows={_vs_rows})"
    except Exception:
        _vs_line = "unknown"
    # CV: gated behind a button; show last result and timestamp
    _cv_score = "n/a"
    try:
        cv_cache = st.session_state.get(Keys.CV_CACHE) or {}
        if isinstance(cv_cache, dict):
            cur = cv_cache.get(str(vm_choice))
            if isinstance(cur, dict) and "acc" in cur:
                acc = float(cur.get("acc", float("nan")))
                k = int(cur.get("k", 0))
                if vm_choice == "XGBoost":
                    _cv_score = f"{acc*100:.0f}% (k={k}, XGB, nested)" if acc == acc else "n/a"
                else:
                    _cv_score = f"{acc*100:.0f}% (k={k})" if acc == acc else "n/a"
    except Exception:
        pass
    # Button to compute CV on demand
    try:
        do_cv = getattr(st.sidebar, "button", lambda *a, **k: False)("Compute CV now")
    except Exception:
        do_cv = False
    if do_cv:
        try:
            from metrics import ridge_cv_accuracy as _rcv, xgb_cv_accuracy as _xcv
            import numpy as _np
            if X_ is not None and y_ is not None and getattr(X_, "shape", (0,))[0] >= 4:
                n_rows = int(len(y_))
                # Ridge CV
                _k_r = min(5, n_rows)
                lam_now = float(st.session_state.get(Keys.REG_LAMBDA, 1e-3))
                acc_r = float(_rcv(X_, y_, lam=lam_now, k=_k_r))
                # XGB CV (uses sidebar hyperparams)
                try:
                    n_estim = int(st.session_state.get("xgb_n_estimators", 50))
                except Exception:
                    n_estim = 50
                try:
                    max_depth = int(st.session_state.get("xgb_max_depth", 3))
                except Exception:
                    max_depth = 3
                try:
                    k_pref = int(st.session_state.get("xgb_cv_folds", 3))
                except Exception:
                    k_pref = 3
                kx = max(2, min(5, min(k_pref, n_rows)))
                acc_x = float(_xcv(X_, y_, k=kx, n_estimators=n_estim, max_depth=max_depth))
                cc = {
                    "Ridge": {"acc": acc_r, "k": _k_r},
                    "XGBoost": {"acc": acc_x, "k": kx},
                }
                st.session_state[Keys.CV_CACHE] = cc
                from datetime import datetime, timezone
                st.session_state[Keys.CV_LAST_AT] = datetime.now(timezone.utc).isoformat(timespec='seconds')
                # Update display for the active VM immediately
                if vm_choice == "XGBoost":
                    _cv_score = f"{acc_x*100:.0f}% (k={kx}, XGB, nested)" if not _np.isnan(acc_x) else "n/a"
                else:
                    _cv_score = f"{acc_r*100:.0f}% (k={_k_r})" if not _np.isnan(acc_r) else "n/a"
        except Exception:
            pass
    # Value model type and settings (use dropdown choice if present)
    try:
        cache = st.session_state.get('xgb_cache') or {}
        if vm_choice == "XGBoost" or (cache.get('model') is not None and vm_choice not in ("DistanceHill","CosineHill")):
            _vm_type = "XGBoost"
            _vm_settings = "default"
        elif vm_choice == "DistanceHill":
            _vm_type = "DistanceHill"
            _vm_settings = "γ=0.5"
        elif vm_choice == "CosineHill":
            _vm_type = "CosineHill"
            _vm_settings = "β=5.0"
        else:
            _vm_type = "Ridge"
            _vm_settings = f"λ={float(st.session_state.get('reg_lambda', 1e-3)):.3g}"
    except Exception:
        _vm_type, _vm_settings = "Ridge", "λ=1e-3"

    # Keep Train score alongside rows; render as a simple metric to avoid
    # tying it to the fragment refresh cadence.
    sidebar_metric_rows([( "Train score", _train_score)], per_row=1)
    # Dimension-scoped rows only (simplify sidebar)
    try:
        from persistence import dataset_rows_for_prompt_dim as _rows_dim
        rows_this_d = _rows_dim(base_prompt, int(getattr(lstate, 'd', 0)))
        sidebar_metric_rows([("Rows (this d)", rows_this_d)], per_row=1)
    except Exception:
        pass
    try:
        min_train_interval = getattr(st.sidebar, "number_input", _sb_num)(
            "Min seconds between trains",
            value=float(st.session_state.get("min_train_interval_s", 0.0)), step=1.0, format="%.0f"
        )
        st.session_state["min_train_interval_s"] = float(min_train_interval)
    except Exception:
        pass
    # Toggle: retrain when new labels are written (default: on)
    try:
        tr_cb = getattr(st.sidebar, "checkbox", lambda *a, **k: True)(
            "Train on new data", value=bool(st.session_state.get('train_on_new_data', True))
        )
        st.session_state['train_on_new_data'] = bool(tr_cb)
    except Exception:
        pass
    # Async training toggles
    try:
        ridge_async_cb = getattr(st.sidebar, "checkbox", lambda *a, **k: st.session_state.get(Keys.RIDGE_TRAIN_ASYNC, False))(
            "Train Ridge async", value=bool(st.session_state.get(Keys.RIDGE_TRAIN_ASYNC, False))
        )
        st.session_state[Keys.RIDGE_TRAIN_ASYNC] = bool(ridge_async_cb)
    except Exception:
        pass
    # XGBoost-only settings: simple hyperparams + CV folds to trade noise vs runtime.
    if vm_choice == "XGBoost":
        try:
            async_train_cb = getattr(st.sidebar, "checkbox", lambda *a, **k: st.session_state.get("xgb_train_async", True))(
                "Train XGBoost async", value=bool(st.session_state.get("xgb_train_async", True))
            )
            st.session_state["xgb_train_async"] = bool(async_train_cb)
        except Exception:
            pass
        try:
            n_estim = int(_sb_num("XGB n_estimators",
                                  value=int(st.session_state.get("xgb_n_estimators", 50)), step=10))
            st.session_state["xgb_n_estimators"] = n_estim
        except Exception:
            pass
        try:
            max_depth = int(_sb_num("XGB max_depth",
                                    value=int(st.session_state.get("xgb_max_depth", 3)), step=1))
            st.session_state["xgb_max_depth"] = max_depth
        except Exception:
            pass
        try:
            k_pref = int(_sb_num("CV folds (XGB)",
                                 value=int(st.session_state.get("xgb_cv_folds", 3)), step=1))
            st.session_state["xgb_cv_folds"] = k_pref
        except Exception:
            pass
    else:
        # Hide async toggle in non-XGB modes to reduce clutter
        st.session_state["xgb_train_async"] = False
    try:
        _last_cv = str(st.session_state.get(Keys.CV_LAST_AT)) if st.session_state.get(Keys.CV_LAST_AT) else 'n/a'
    except Exception:
        _last_cv = 'n/a'
    # Group training results to keep the sidebar tidy
    _exp_tr = getattr(st.sidebar, 'expander', None)
    if callable(_exp_tr):
        try:
            with _exp_tr("Train results", expanded=False):
                try:
                    st.sidebar.write(f"Train score: {_train_score}")
                except Exception:
                    pass
                try:
                    st.sidebar.write(f"CV score: {_cv_score}")
                    st.sidebar.write(f"Last CV: {_last_cv}")
                except Exception:
                    pass
                try:
                    st.sidebar.write(f"Last train: {_last_train}")
                except Exception:
                    pass
                try:
                    st.sidebar.write(f"Value scorer: {_vs_line}")
                except Exception:
                    pass
                # Minimal Ridge status line (running/ok/idle)
                try:
                    import numpy as _np
                    fut_r = st.session_state.get(Keys.RIDGE_FIT_FUTURE)
                    running_r = bool(fut_r is not None and not getattr(fut_r, "done", lambda: False)())
                    w_now = getattr(lstate, "w", None)
                    wn = float(_np.linalg.norm(w_now)) if w_now is not None else 0.0
                    status_r = "running" if running_r else ("ok" if wn > 0.0 else "idle")
                    st.sidebar.write(f"Ridge training: {status_r}")
                    if fut_r is not None and getattr(fut_r, "done", lambda: False)():
                        st.session_state.pop(Keys.RIDGE_FIT_FUTURE, None)
                except Exception:
                    pass
        except TypeError:
            st.sidebar.write(f"Train score: {_train_score}")
            st.sidebar.write(f"CV score: {_cv_score}")
            st.sidebar.write(f"Last CV: {_last_cv}")
            st.sidebar.write(f"Last train: {_last_train}")
            st.sidebar.write(f"Value scorer: {_vs_line}")
            try:
                import numpy as _np
                fut_r = st.session_state.get(Keys.RIDGE_FIT_FUTURE)
                running_r = bool(fut_r is not None and not getattr(fut_r, "done", lambda: False)())
                w_now = getattr(lstate, "w", None)
                wn = float(_np.linalg.norm(w_now)) if w_now is not None else 0.0
                status_r = "running" if running_r else ("ok" if wn > 0.0 else "idle")
                st.sidebar.write(f"Ridge training: {status_r}")
                if fut_r is not None and getattr(fut_r, "done", lambda: False)():
                    st.session_state.pop(Keys.RIDGE_FIT_FUTURE, None)
            except Exception:
                pass
    else:
        st.sidebar.write(f"Train score: {_train_score}")
        st.sidebar.write(f"CV score: {_cv_score}")
        st.sidebar.write(f"Last CV: {_last_cv}")
        st.sidebar.write(f"Last train: {_last_train}")
        st.sidebar.write(f"Value scorer: {_vs_line}")
        try:
            import numpy as _np
            fut_r = st.session_state.get(Keys.RIDGE_FIT_FUTURE)
            running_r = bool(fut_r is not None and not getattr(fut_r, "done", lambda: False)())
            w_now = getattr(lstate, "w", None)
            wn = float(_np.linalg.norm(w_now)) if w_now is not None else 0.0
            status_r = "running" if running_r else ("ok" if wn > 0.0 else "idle")
            st.sidebar.write(f"Ridge training: {status_r}")
            if fut_r is not None and getattr(fut_r, "done", lambda: False)():
                st.session_state.pop(Keys.RIDGE_FIT_FUTURE, None)
        except Exception:
            pass
    # Tiny visibility line for XGBoost readiness
    try:
        if vm_choice == "XGBoost":
            fut = st.session_state.get(Keys.XGB_FIT_FUTURE)
            fut_running = bool(fut is not None and not getattr(fut, "done", lambda: False)())
            active = "yes" if (_vs_status == "ok") else "no"
            status = st.session_state.get(Keys.XGB_TRAIN_STATUS)
            st.sidebar.write(f"XGBoost active: {active}")
            if isinstance(status, dict):
                state = status.get("state")
                rows = status.get("rows")
                lam = status.get("lam")
                if state == "running" or fut_running:
                    st.sidebar.write(f"Train progress: rows={rows} λ={lam}")
                elif state == "waiting":
                    st.sidebar.write("Train progress: waiting (cooldown)")
                elif state == "ok":
                    st.sidebar.write(f"Updated XGB (rows={rows}, λ={lam})")
            # If a background fit just finished, clear the future; we avoid full page rerun by default.
            if fut is not None and getattr(fut, "done", lambda: False)():
                st.session_state.pop(Keys.XGB_FIT_FUTURE, None)
    except Exception:
        pass
    # Tiny hint when XGBoost is selected but no model is trained yet.
    try:
        if vm_choice == "XGBoost" and "xgb_unavailable" in _vs_line:
            st.sidebar.write("Hint: XGBoost not trained yet – label a few images (Good/Bad) to fit it.")
    except Exception:
        pass
    # Place editable λ above the model metrics
    try:
        if _vm_type == "Ridge":
            lam_top = _sb_num("Ridge λ (edit)",
                              value=float(st.session_state.get('reg_lambda', 0.0)),
                              step=1e-3, format="%.6f")
            st.session_state['reg_lambda'] = float(lam_top)
    except Exception:
        pass
    sidebar_metric_rows([( "Value model", _vm_type), ("Settings", _vm_settings)], per_row=2)
    # Keep dim-mismatch notice even when metrics above are shown
    try:
        mm = st.session_state.get("dataset_dim_mismatch")
        if mm:
            st.sidebar.write(
                f"Dataset recorded at different resolution (d={mm[0]}) – "
                f"current latent dim d={mm[1]}; ignoring saved dataset for training."
            )
    except Exception:
        pass
except Exception:
    pass

# Model & decode settings
st.sidebar.header("Model & decode settings")
try:
    # Fragment toggle for image tiles
    use_frags = bool(getattr(st.sidebar, 'checkbox', lambda *a, **k: True)(
        "Use fragments (isolate image tiles)", value=bool(st.session_state.get('use_fragments', True))
    ))
    st.session_state['use_fragments'] = use_frags
    use_srv = bool(getattr(st.sidebar, 'checkbox', lambda *a, **k: False)(
        "Use image server", value=bool(st.session_state.get('use_image_server', False))
    ))
    st.session_state['use_image_server'] = use_srv
    srv_url = getattr(st.sidebar, 'text_input', lambda *a, **k: os.getenv('IMAGE_SERVER_URL', ''))(
        "Image server URL", value=str(st.session_state.get('image_server_url', os.getenv('IMAGE_SERVER_URL', '')))
    )
    st.session_state['image_server_url'] = srv_url
    try:
        import flux_local as _fl
        _uis = getattr(_fl, 'use_image_server', None)
        if callable(_uis):
            _uis(use_srv, srv_url)
        # Optional health check when server is enabled
        if use_srv and srv_url:
            import json as _json
            import urllib.request as _url
            try:
                with _url.urlopen(srv_url.rstrip('/') + '/health', timeout=2) as r:  # nosec - user-provided URL
                    ok = bool(_json.loads(r.read().decode('utf-8')).get('ok'))
                st.sidebar.write(f"Image server: {'ok' if ok else 'unavailable'}")
            except Exception:
                st.sidebar.write("Image server: unavailable")
    except Exception:
        pass
except Exception:
    pass
width, height, steps, guidance, _apply_clicked = build_size_controls(st, lstate)
try:
    st.session_state['steps'] = int(steps)
    st.session_state['guidance'] = float(guidance)
except Exception:
    pass
if _apply_clicked:
    _apply_state(init_latent_state(width=int(width), height=int(height)))
    save_state(st.session_state.lstate, st.session_state.state_path)
    _toast(f"Applied size {int(width)}x{int(height)}")
    if callable(st_rerun):
        st_rerun()
_model_sel = getattr(st.sidebar, 'selectbox', None)
if callable(_model_sel):
    try:
        selected_model = _model_sel("Model", MODEL_CHOICES, index=0)
    except Exception:
        selected_model = DEFAULT_MODEL
else:
    selected_model = DEFAULT_MODEL

## Generation mode moved to top-of-sidebar (see "Mode" section above)
# Proposer controls (alpha/beta/etc.) are still rendered for maintainability and tests,
# but they are only used by Async/DistanceHill proposers internally.

def _pair_controls_store_session():
    st.sidebar.subheader("Latent optimization")
    _alpha, _beta, _trust_r, _lr_mu_ui, _gamma_orth, _iter_steps, _iter_eta = build_pair_controls(st, expanded=False)
    for k, v in (('alpha', _alpha), ('trust_r', _trust_r), ('lr_mu_ui', _lr_mu_ui)):
        try:
            st.session_state[k] = float(v)
        except Exception:
            pass
    return {'lr_mu_ui': _lr_mu_ui, 'iter_steps': _iter_steps, 'iter_eta': _iter_eta}


def _hill_climb_controls(nonlocal_vars):
    st.sidebar.subheader("Hill-climb μ")
    _eta_mu = _sb_num("η (step)", value=0.2, step=0.01, format="%.2f")
    _gamma_mu = _sb_num("γ (sigmoid)", value=0.5, step=0.1, format="%.1f")
    _trust_mu = _sb_num("Trust radius r (0=off)", value=0.0, step=1.0, format="%.1f")
    _hill_label = "Hill-climb μ (XGBoost)" if vm_choice == "XGBoost" else "Hill-climb μ (distance)"
    if not st.sidebar.button(_hill_label):
        return
    from persistence import get_dataset_for_prompt_or_session
    Xd, yd = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
    try:
        if Xd is None or yd is None or getattr(Xd, 'shape', (0,))[0] == 0:
            return
        r_val = None if float(_trust_mu) <= 0.0 else float(_trust_mu)
        def _hc_xgb() -> None:
            from value_scorer import get_value_scorer
            cache = st.session_state.get('xgb_cache') or {}
            mdl = cache.get('model')
            if mdl is None:
                return
            scorer = get_value_scorer("XGBoost", lstate, base_prompt, st.session_state)
            step_scale = float(nonlocal_vars['lr_mu_ui']) * float(getattr(lstate, "sigma", 1.0))
            steps_now = int(st.session_state.get("iter_steps", nonlocal_vars['iter_steps']))
            ll.hill_climb_mu_xgb(lstate, base_prompt, scorer, steps=steps_now, step_scale=step_scale, trust_r=r_val)
            save_state(lstate, st.session_state.state_path)

        def _hc_distance() -> None:
            ll.hill_climb_mu_distance(lstate, base_prompt, Xd, yd, eta=float(_eta_mu), gamma=float(_gamma_mu), trust_r=r_val)
            save_state(lstate, st.session_state.state_path)

        (_hc_xgb if vm_choice == "XGBoost" else _hc_distance)()
    except Exception:
        pass
    if callable(st_rerun):
        st_rerun()


def _render_advanced_controls():
    vars_local = _pair_controls_store_session()
    _hill_climb_controls(vars_local)


# In compact mode, tuck advanced controls into an expander; otherwise render inline
_adv_expander = None
try:
    if bool(st.session_state.get('sidebar_compact', False)) and callable(getattr(st.sidebar, 'expander', None)):
        _adv_expander = st.sidebar.expander("Advanced", expanded=False)
except Exception:
    _adv_expander = None

if _adv_expander is not None:
    try:
        with _adv_expander:
            _render_advanced_controls()
    except TypeError:
        _render_advanced_controls()
else:
    _render_advanced_controls()
# Value function option: controlled solely by the dropdown above
use_xgb = (vm_choice == "XGBoost")

# Legacy toggles removed to avoid duplicate mode controls in the sidebar.
curation_mode_cb, async_queue_mode_cb = False, False

try:
    # Best-of toggle applies only to Batch mode; keep state for tests when mode is unset.
    if selected_gen_mode is None or selected_gen_mode == _gen_opts[0]:
        best_of = bool(getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
            "Best-of batch (one winner)", value=bool(getattr(st.session_state, "batch_best_of", False))
        ))
        st.session_state['batch_best_of'] = best_of
except Exception:
    pass

def _resolve_modes():
    """Return (curation_mode, async_queue_mode) from dropdown/checkboxes."""
    if selected_gen_mode is not None:
        return (selected_gen_mode == _gen_opts[0], selected_gen_mode == _gen_opts[1])
    return (bool(curation_mode_cb), bool(async_queue_mode_cb))

curation_mode, async_queue_mode = _resolve_modes()
# Regularization λ: single precise input (slider removed)
try:
    reg_lambda = float(_sb_num("Ridge λ", value=1e-3, step=1e-3, format="%.6f"))
except Exception:
    reg_lambda = 1e-3
try:
    st.session_state['reg_lambda'] = float(reg_lambda)
except Exception:
    pass
try:
    eta_default = float(getattr(st.session_state, "iter_eta", 0.1))
except Exception:
    eta_default = 0.1
iter_eta_num = _sb_num("Iterative step (eta)", value=eta_default, step=0.01, format="%.2f")
try:
    st.session_state["iter_eta"] = float(iter_eta_num)
except Exception:
    pass
try:
    iter_eta = float(getattr(st.session_state, "iter_eta", iter_eta_num))
except Exception:
    iter_eta = float(iter_eta_num)

# Numeric control for Optimization steps (latent); overrides any prior default.
try:
    from constants import DEFAULT_ITER_STEPS as _DEF_STEPS
except Exception:
    _DEF_STEPS = 10
try:
    steps_default = int(getattr(st.session_state, "iter_steps", _DEF_STEPS))
except Exception:
    steps_default = int(_DEF_STEPS)
iter_steps_num = _sb_num("Optimization steps (latent)", value=steps_default, step=1)
try:
    st.session_state["iter_steps"] = int(iter_steps_num)
except Exception:
    pass
try:
    iter_steps = int(getattr(st.session_state, "iter_steps", iter_steps_num))
except Exception:
    iter_steps = int(iter_steps_num)
use_clip = False

# 7 GB VRAM recipe: lighter model, smaller size, no CLIP
# 7 GB VRAM mode removed; users can lower size/steps manually via controls

is_turbo = True
guidance_eff = 0.0
try:
    st.session_state['guidance_eff'] = float(guidance_eff)
except Exception:
    pass

# (auto-run added after function definitions below)

mu_show = False

def _sidebar_persistence_section():
    st.sidebar.subheader("State persistence")
    _export_state_bytes = export_state_bytes  # back-compat for tests
    render_persistence_controls(lstate, st.session_state.prompt, st.session_state.state_path, _apply_state, st_rerun)


def _render_iter_step_scores_block():
    try:
        _tr = st.session_state.get('trust_r', 0.0)
        trust_val = float(_tr) if (_tr is not None and float(_tr) > 0.0) else None
    except Exception:
        trust_val = None
    render_iter_step_scores(st, lstate, base_prompt, vm_choice, int(iter_steps), float(iter_eta) if iter_eta is not None else None, trust_val)
    render_mu_value_history(st, lstate, base_prompt)


def _ensure_sidebar_shims():
    st.sidebar.subheader("Latent state")
    if not hasattr(st.sidebar, 'write'):
        st.sidebar.text = getattr(st.sidebar, 'text', lambda *a, **k: None)
        def _w(x):
            st.sidebar.text(str(x))
        st.sidebar.write = _w
    if not hasattr(st.sidebar, 'metric'):
        st.sidebar.metric = lambda label, value, **k: st.sidebar.write(f"{label}: {value}")


def _sidebar_training_data_block():
    try:
        exp = getattr(st.sidebar, 'expander', None)
        stats = dataset_stats_for_prompt(base_prompt)
        if callable(exp):
            with exp("Training data", expanded=False):
                sidebar_metric_rows([("Pos", stats.get("pos", 0)), ("Neg", stats.get("neg", 0))], per_row=2)
                sidebar_metric_rows([("Feat dim", stats.get("d", 0))], per_row=1)
                rl = stats.get("recent_labels", [])
                if rl:
                    st.sidebar.write("Recent y: " + ", ".join([f"{v:+d}" for v in rl]))
        else:
            st.sidebar.write("Training data: pos={} neg={} d={}".format(stats.get("pos",0), stats.get("neg",0), stats.get("d",0)))
    except Exception:
        pass


def _sidebar_value_model_block():
    def _sb_w(line: str) -> None:
        try:
            if hasattr(st, 'sidebar_writes'):
                st.sidebar_writes.append(str(line))
            else:
                st.sidebar.write(str(line))
        except Exception:
            pass

    def _cached_cv_lines() -> tuple[str, str]:
        ridge_line = "CV (Ridge): n/a"
        xgb_line = "CV (XGBoost): n/a"
        try:
            cv_cache = st.session_state.get(Keys.CV_CACHE) or {}
            if isinstance(cv_cache, dict):
                r = cv_cache.get("Ridge") or {}
                x = cv_cache.get("XGBoost") or {}
                if "acc" in r and "k" in r:
                    ridge_line = f"CV (Ridge): {float(r['acc'])*100:.0f}% (k={int(r['k'])})"
                if "acc" in x and "k" in x:
                    xgb_line = f"CV (XGBoost): {float(x['acc'])*100:.0f}% (k={int(x['k'])})"
        except Exception:
            pass
        return xgb_line, ridge_line

    def _vm_header_and_status() -> tuple[str, str, dict]:
        vm = "Ridge"
        cache = st.session_state.get('xgb_cache') or {}
        try:
            if use_xgb and cache.get('model') is not None:
                vm = "XGBoost"
        except Exception:
            pass
        try:
            from value_scorer import get_value_scorer_with_status
            _scorer_vm, scorer_status = get_value_scorer_with_status(vm_choice, lstate, base_prompt, st.session_state)
        except Exception:
            scorer_status = "unknown"
        st.sidebar.write(f"Value model: {vm}")
        st.sidebar.write(f"Value scorer status: {scorer_status}")
        return vm, scorer_status, cache

    def _vm_details(vm: str, cache: dict) -> None:
        subexp = getattr(st.sidebar, 'expander', None)
        if not callable(subexp):
            return
        with subexp("Details", expanded=False):
            if vm == "Ridge":
                try:
                    w_norm = float(np.linalg.norm(lstate.w))
                except Exception:
                    w_norm = 0.0
                try:
                    rows = int(dataset_rows_for_prompt(base_prompt))
                except Exception:
                    rows = 0
                st.sidebar.write(f"λ={reg_lambda:.3g}, ||w||={w_norm:.3f}, rows={rows}")
            else:
                n_fit = cache.get('n') or 0
                try:
                    n_estim = int(st.session_state.get("xgb_n_estimators", 50))
                except Exception:
                    n_estim = 50
                try:
                    max_depth = int(st.session_state.get("xgb_max_depth", 3))
                except Exception:
                    max_depth = 3
                st.sidebar.write(f"fit_rows={int(n_fit)}, n_estimators={n_estim}, depth={max_depth}")

    # Always emit cached CV lines so tests/stubs see them
    xgb_line, ridge_line = _cached_cv_lines()
    _sb_w(xgb_line)
    _sb_w(ridge_line)

    exp = getattr(st.sidebar, 'expander', None)
    if not callable(exp):
        _sb_w("Value model: Ridge")
        xgb_line, ridge_line = _cached_cv_lines()
        _sb_w(xgb_line)
        _sb_w(ridge_line)
        return

    with exp("Value model", expanded=False):
        vm, _sc_status, cache = _vm_header_and_status()
        xgb_line, ridge_line = _cached_cv_lines()
        _sb_w(xgb_line)
        _sb_w(ridge_line)
        _vm_details(vm, cache)


def render_sidebar_tail():
    _sidebar_persistence_section()
    _render_iter_step_scores_block()
    # Autorun: set model once so decode paths are ready
    set_model(selected_model)
    _ensure_sidebar_shims()
    _sidebar_training_data_block()
    _sidebar_value_model_block()

# Render the sidebar tail now that mode and controls are resolved
render_sidebar_tail()

## imports moved to top


def generate_pair():
    try:
        _log("[pair] generate_pair() called")
    except Exception:
        pass
    _pair_generate()
    try:
        imgs = st.session_state.get('images')
        if not imgs or imgs[0] is None or imgs[1] is None:
            # Minimal fallback for test stubs: use text-only path if available
            if callable(generate_flux_image):
                img = generate_flux_image(base_prompt, width=lstate.width, height=lstate.height, steps=Config.DEFAULT_STEPS, guidance=Config.DEFAULT_GUIDANCE)
                st.session_state.images = (img, img)
    except Exception:
        pass


def _prefetch_next_for_generate():
    _pair_prefetch()

    # μ preview removed


# history helpers removed

def _curation_init_batch() -> None:
    return _batch_ui._curation_init_batch()


def _curation_new_batch() -> None:
    return _batch_ui._curation_new_batch()


def _sample_around_prompt(scale: float = 0.8) -> np.ndarray:
    return _batch_ui._sample_around_prompt(scale)


def _curation_replace_at(idx: int) -> None:
    return _batch_ui._curation_replace_at(idx)


def _proposer_opts():
    """Return a ProposerOpts built from current sidebar settings (delegates)."""
    from proposer import build_proposer_opts
    # Read from session to avoid free-name leaks when controls are hidden
    try:
        it_steps = int(st.session_state.get("iter_steps", 10))
    except Exception:
        it_steps = 10
    try:
        ie = st.session_state.get("iter_eta", None)
        ie = float(ie) if ie is not None else None
    except Exception:
        ie = None
    try:
        tr = st.session_state.get("trust_r", None)
        tr = float(tr) if (tr is not None and float(tr) > 0.0) else None
    except Exception:
        tr = None
    try:
        g = float(st.session_state.get("gamma_orth", 0.0))
    except Exception:
        g = 0.0
    return build_proposer_opts(int(it_steps), ie, tr, g)


def _curation_add(label: int, z: np.ndarray, img=None) -> None:
    return _batch_ui._curation_add(label, z, img)


def _curation_train_and_next() -> None:
    return _batch_ui._curation_train_and_next()


def _refit_from_dataset_keep_batch() -> None:
    """Refit ridge from saved dataset (or in-memory) without regenerating the batch."""
    import streamlit as _st
    from persistence import get_dataset_for_prompt_or_session
    X, y = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
    try:
        if X is not None and y is not None and getattr(X, 'shape', (0,))[0] > 0:
            lam_now = float(getattr(_st.session_state, 'reg_lambda', reg_lambda))
            import value_model as _vm
            tr = getattr(_vm, 'train_and_record', None)
            vmc = _st.session_state.get('vm_choice')
            if callable(tr):
                tr(vmc, lstate, X, y, lam_now, _st.session_state)
            else:
                fit_value_model(vmc, lstate, X, y, lam_now, _st.session_state)
    except Exception:
        pass


def _label_and_persist(z: np.ndarray, label: int, retrain: bool = True) -> None:
    """Unified label/save/train pipeline for Pair/Batch/Queue."""
    _curation_add(int(label), z)
    if retrain:
        try:
            _curation_train_and_next()
        except Exception:
            pass


def _queue_fill_up_to() -> None:
    import queue_ui as _q
    return _q._queue_fill_up_to()


def _queue_label(idx: int, label: int) -> None:
    import queue_ui as _q
    return _q._queue_label(idx, label)


def _choose_preference(side: str) -> None:
    """Handle pair preference: update state, persist both labels, propose next pair."""
    import streamlit as _st
    set_model(selected_model)
    z_p = z_from_prompt(lstate, base_prompt)
    feats_a = z_a - z_p
    feats_b = z_b - z_p
    winner = 'a' if side == 'a' else 'b'
    lam_now = float(getattr(_st.session_state, 'reg_lambda', reg_lambda))
    try:
        lr_mu_now = float(_st.session_state.get('lr_mu_ui', 0.3))
    except Exception:
        lr_mu_now = 0.3
    update_latent_ridge(lstate, z_a, z_b, winner, lr_mu=lr_mu_now, lam=lam_now, feats_a=feats_a, feats_b=feats_b)
    try:
        from datetime import datetime, timezone
        _st.session_state['last_train_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    except Exception:
        pass
    try:
        _label_and_persist(z_a, +1 if winner == 'a' else -1)
        _label_and_persist(z_b, +1 if winner == 'b' else -1)
    except Exception:
        pass
    st.session_state.lz_pair = propose_next_pair(lstate, base_prompt, opts=_proposer_opts())
    save_state(lstate, st.session_state.state_path)
    if callable(st_rerun):
        st_rerun()


## Pair UI renderer removed (pair mode no longer routed); generate_pair remains for tests.


def _render_batch_ui() -> None:
    return _batch_ui._render_batch_ui()


## Queue UI renderer imported from queue_ui for test compatibility


def _pair_scores() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    return _pair_scores_impl()


## Deprecated helpers removed; tests should use queue_ui/batch_ui directly


## Pair mode runner removed; only Batch and Queue are routed.


def run_batch_mode() -> None:
    try:
        _log("[mode] running Batch curation")
    except Exception:
        pass
    return _batch_ui.run_batch_mode()


def run_upload_mode() -> None:
    st.subheader("Upload latents")
    lstate, prompt = _batch_ui._lstate_and_prompt()
    from value_scorer import get_value_scorer_with_status
    scorer, scorer_status = get_value_scorer_with_status(
        st.session_state.get("vm_choice"), lstate, prompt, st.session_state
    )

    def _get_uploads_and_params():
        uploads = getattr(st.sidebar, "file_uploader", lambda *a, **k: [])(
            "Upload images to use as latents", accept_multiple_files=True, type=["png", "jpg", "jpeg", "webp"]
        )
        steps = int(getattr(st.session_state, "steps", 6))
        guidance_eff = float(getattr(st.session_state, "guidance_eff", 0.0))
        z_p_local = z_from_prompt(lstate, prompt)
        nonce_local = int(st.session_state.get("cur_batch_nonce", 0))
        return uploads, steps, guidance_eff, z_p_local, nonce_local

    def _save_upload_image(img_raw, nonce_local: int, idx: int) -> None:
        try:
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
            up_dir = os.path.join("data", h, "uploads")
            os.makedirs(up_dir, exist_ok=True)
            fname = f"upload_{nonce_local}_{idx}.png"
            img_raw.save(os.path.join(up_dir, fname))
        except Exception:
            pass

    def _process_upload(idx: int, upl, steps: int, guidance_eff: float, z_p_local, nonce_local: int):
        try:
            img_raw = Image.open(upl)
        except Exception:
            return
        _save_upload_image(img_raw, nonce_local, idx)
        z_upl = image_to_z(img_raw, lstate)
        alpha_interp = st.slider(
            f"Interpolate toward prompt (α) {idx}",
            value=1.0, step=0.05, key=f"upl_interp_{nonce_local}_{idx}"
        )
        z = (1.0 - float(alpha_interp)) * z_p_local + float(alpha_interp) * z_upl
        try:
            lat = z.reshape(1, 4, lstate.height // 8, lstate.width // 8)
        except Exception:
            return
        img_dec = generate_flux_image_latents(
            prompt, latents=lat, width=lstate.width, height=lstate.height,
            steps=steps, guidance=guidance_eff,
        )
        st.image(img_dec, caption=f"Upload {idx}", width="stretch")
        try:
            if scorer is not None and scorer_status == "ok":
                score_val = float(scorer(z - z_p_local))
                st.caption(f"Score: {score_val:.3f}")
            else:
                st.caption("Score: n/a")
        except Exception:
            pass
        w = st.slider(
            f"Weight upload {idx}", value=1.0,
            step=0.1, key=f"upl_w_{nonce_local}_{idx}"
        )
        if st.button(f"Good (+1) upload {idx}", key=f"upl_good_{nonce_local}_{idx}"):
            _curation_add(float(w), z, img=None)
            _curation_train_and_next()
        if st.button(f"Bad (-1) upload {idx}", key=f"upl_bad_{nonce_local}_{idx}"):
            _curation_add(-float(w), z, img=None)
            _curation_train_and_next()

    uploads, steps, guidance_eff, z_p, nonce = _get_uploads_and_params()
    if not uploads:
        st.write("Upload at least one image to score it as Good/Bad.")
        return
    for idx, upl in enumerate(uploads):
        _process_upload(idx, upl, steps, guidance_eff, z_p, nonce)


## import moved to top
# Run selected mode (Batch default vs Async queue)
try:
    async_queue_mode
except NameError:  # minimal guard for test stubs/import order
    async_queue_mode = False
try:
    _log(f"[mode] dispatch async_queue_mode={bool(async_queue_mode)}")
except Exception:
    pass
if selected_gen_mode == _gen_opts[2]:
    run_upload_mode()
else:
    run_mode(bool(async_queue_mode))

st.write(f"Interactions: {lstate.step}")
if st.button("Reset", type="secondary"):
    _apply_state(init_latent_state(width=int(width), height=int(height)))
    save_state(st.session_state.lstate, st.session_state.state_path)
    if callable(st_rerun):
        st_rerun()

st.caption(f"Persistence: {st.session_state.state_path}{' (loaded)' if os.path.exists(st.session_state.state_path) else ''}")
# Footer: recent prompt states (hash + truncated text)
recent = st.session_state.get('recent_prompts', [])
if recent:
    def _hash_of(p: str) -> str:
        return hashlib.sha1(p.encode('utf-8')).hexdigest()[:10]
    items = [f"{_hash_of(p)} • {p[:30]}" for p in recent[:3]]
    st.caption("Recent states: " + ", ".join(items))

## First-round prompt seeding is handled in _apply_state; no duplicate logic here

    # μ preview UI removed
## _toast already defined above; avoid duplicate definitions
