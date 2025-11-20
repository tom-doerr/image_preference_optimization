import streamlit as st
import numpy as np
from typing import Optional, Tuple, Any
import os
import hashlib
from PIL import Image
from PIL import Image
from constants import (
    DEFAULT_PROMPT,
    DEFAULT_MODEL,
    MODEL_CHOICES,
)
from constants import Config
from env_info import get_env_summary
from ui import sidebar_metric_rows, render_pair_sidebar, env_panel, status_panel, perf_panel
from concurrent import futures  # exposed for tests to monkey-patch executors
import batch_ui as _batch_ui
from batch_ui import (
    _curation_init_batch,
    _curation_new_batch,
    _curation_replace_at,
    _curation_add,
    _curation_train_and_next,
    _refit_from_dataset_keep_batch,
    _render_batch_ui,
    run_batch_mode,
)
from queue_ui import (
    _queue_fill_up_to,
    _queue_label,
    _render_queue_ui,
    run_queue_mode,
)
from persistence import state_path_for_prompt, export_state_bytes, dataset_path_for_prompt, dataset_rows_for_prompt, dataset_stats_for_prompt
import background as bg
from persistence_ui import render_persistence_controls, render_metadata_panel, render_paths_panel, render_dataset_viewer
from latent_opt import (
    init_latent_state,
    propose_next_pair,
    z_to_latents,
    z_from_prompt,
    update_latent_ridge,
    save_state,
    load_state,
    state_summary,
    propose_latent_pair_ridge,
)
from value_model import fit_value_model
import latent_logic as ll  # module alias for patchable ridge_fit in tests
from flux_local import (
    generate_flux_image_latents,
    set_model,
)
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
if callable(_frag):
    _image_fragment = _frag(_image_fragment)

def _apply_state(new_state) -> None:
    """Apply a freshly loaded/created state to session and reset derived caches."""
    st.session_state.lstate = new_state
    # Carry over optional random-anchor flag from session into the new state
    try:
        use_rand = bool(getattr(st.session_state, "use_random_anchor", False))
        setattr(new_state, "use_random_anchor", use_rand)
        setattr(new_state, "random_anchor_z", None)
    except Exception:
        pass
    # Initialize pair around the prompt anchor (symmetric)
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
st.session_state['vm_choice'] = vm_choice
# Training model selector (independent so you can fit Ridge while scoring with Distance/Cosine, etc.)
_vm_train_opts = ["XGBoost", "Ridge"]
vm_train_choice = None
if callable(_sb_sel):
    try:
        vm_train_choice = _sb_sel("Train value model", _vm_train_opts, index=0)
        if vm_train_choice not in _vm_train_opts:
            vm_train_choice = "XGBoost"
    except Exception:
        vm_train_choice = "XGBoost"
else:
    vm_train_choice = "XGBoost"
st.session_state['vm_train_choice'] = vm_train_choice

# Batch/queue controls near top for quick access
_exp = getattr(st.sidebar, 'expander', None)
from ui_controls import build_batch_controls, build_queue_controls
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
        st.session_state['queue_size'] = int(queue_size)
except Exception:
    pass
try:
    st.session_state['batch_size'] = int(batch_size)
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

# Quick data strip
try:
    st.sidebar.subheader("Training data & scores")
    # Dataset rows from persisted NPZ
    try:
        _rows_cnt = int(dataset_rows_for_prompt(base_prompt))
    except Exception:
        _rows_cnt = 0
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
                _ensure_fitted(st.session_state.get('vm_train_choice', vm_choice), lstate, X_, y_, lam_auto, st.session_state)
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
    try:
        from metrics import ridge_cv_accuracy as _rcv, xgb_cv_accuracy as _xcv
        if X_ is not None and y_ is not None and len(y_) >= 4:
            n_rows = int(len(y_))
            if vm_choice == "XGBoost":
                try:
                    import numpy as _np
                    # Read XGB hyperparams for CV to mirror training settings.
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
                    k = max(2, min(5, min(k_pref, n_rows)))
                    _cv = float(_xcv(X_, y_, k, n_estimators=n_estim, max_depth=max_depth))
                    if _np.isnan(_cv):
                        _cv_score = "n/a"
                    else:
                        _cv_score = f"{_cv*100:.0f}% (k={k}, XGB, nested)"
                except Exception:
                    _cv_score = "n/a"
            else:
                _k = min(5, n_rows)
                _cv = float(_rcv(X_, y_, lam=float(st.session_state.get('reg_lambda', 1e-3)), k=_k))
                _cv_score = f"{_cv*100:.0f}% (k={_k})"
        else:
            _cv_score = "n/a"
    except Exception:
        _cv_score = "n/a"
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

    sidebar_metric_rows([("Dataset rows", _rows_cnt), ("Train score", _train_score)], per_row=2)
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
            n_estim = int(_sb_num("XGB n_estimators", min_value=10, max_value=200,
                                  value=int(st.session_state.get("xgb_n_estimators", 50)), step=10))
            st.session_state["xgb_n_estimators"] = n_estim
        except Exception:
            pass
        try:
            max_depth = int(_sb_num("XGB max_depth", min_value=2, max_value=8,
                                    value=int(st.session_state.get("xgb_max_depth", 3)), step=1))
            st.session_state["xgb_max_depth"] = max_depth
        except Exception:
            pass
        try:
            k_pref = int(_sb_num("CV folds (XGB)", min_value=2, max_value=5,
                                 value=int(st.session_state.get("xgb_cv_folds", 3)), step=1))
            st.session_state["xgb_cv_folds"] = k_pref
        except Exception:
            pass
    else:
        # Hide async toggle in non-XGB modes to reduce clutter
        st.session_state["xgb_train_async"] = False
    sidebar_metric_rows([("CV score", _cv_score), ("Last train", _last_train)], per_row=2)
    sidebar_metric_rows([("Value scorer", _vs_line)], per_row=1)
    # Tiny visibility line for XGBoost readiness
    try:
        if vm_choice == "XGBoost":
            fut = st.session_state.get("xgb_fit_future")
            fut_running = bool(fut is not None and not getattr(fut, "done", lambda: False)())
            active = "yes" if (_vs_status == "ok") else "no"
            status = st.session_state.get("xgb_train_status")
            if fut_running or (isinstance(status, dict) and status.get("state") == "running"):
                st.sidebar.write("XGBoost active: training…")
                if isinstance(status, dict) and status.get("state") == "running":
                    rows = status.get("rows")
                    lam = status.get("lam")
                    st.sidebar.write(f"Train progress: rows={rows} λ={lam}")
            else:
                st.sidebar.write(f"XGBoost active: {active}")
                if isinstance(status, dict) and status.get("state") == "ok":
                    rows = status.get("rows")
                    lam = status.get("lam")
                    st.sidebar.write(f"Train progress: updated (rows={rows}, λ={lam})")
                # One-shot toast/note after fit completion
                try:
                    last_rows = st.session_state.pop("xgb_last_updated_rows", None)
                    if last_rows is not None:
                        st.sidebar.write(f"Updated XGB (rows={last_rows})")
                        try:
                            st.sidebar.success(f"Updated XGB (rows={last_rows})")
                        except Exception:
                            pass
                        st.session_state.pop("xgb_last_updated_lam", None)
                except Exception:
                    pass
            # If a background fit just finished, clear the future; we avoid full page rerun by default.
            if fut is not None and getattr(fut, "done", lambda: False)():
                st.session_state.pop("xgb_fit_future", None)
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
            lam_top = _sb_num("Ridge λ (edit)", min_value=0.0, max_value=1e5,
                              value=float(st.session_state.get('reg_lambda', 0.0)),
                              step=1e-3, format="%.6f")
            st.session_state['reg_lambda'] = float(lam_top)
    except Exception:
        pass
    sidebar_metric_rows([( "Value model", _vm_type), ("Settings", _vm_settings)], per_row=2)
    # Ensure visibility in simple stubs that only capture writes
    try:
        st.sidebar.write(f"Dataset rows: {_rows_cnt}")
        st.sidebar.write(f"Train score: {_train_score}")
        st.sidebar.write(f"Value model: {_vm_type}")
        st.sidebar.write(f"Settings: {_vm_settings}")
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
from ui_controls import build_size_controls
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
from constants import DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_TRUST_R, DEFAULT_LR_MU, DEFAULT_GAMMA_ORTH, DEFAULT_ITER_STEPS
from ui_controls import build_pair_controls

# Optimization controls
st.sidebar.subheader("Latent optimization")
# Proposer controls (Alpha/Beta/etc.) shared across modes; implemented via a single helper.
alpha, beta, trust_r, lr_mu_ui, gamma_orth, iter_steps, iter_eta = build_pair_controls(st, expanded=False)
try:
    st.session_state['alpha'] = float(alpha)
except Exception:
    pass
try:
    st.session_state['trust_r'] = float(trust_r)
except Exception:
    pass
try:
    st.session_state['lr_mu_ui'] = float(lr_mu_ui)
except Exception:
    pass

# Hill-climb μ controls (shared across modes)
st.sidebar.subheader("Hill-climb μ")
eta_mu = _sb_num("η (step)", min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f")
gamma_mu = _sb_num("γ (sigmoid)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, format="%.1f")
trust_mu = _sb_num("Trust radius r (0=off)", min_value=0.0, max_value=1e3, value=0.0, step=1.0, format="%.1f")
hill_label = "Hill-climb μ (XGBoost)" if vm_choice == "XGBoost" else "Hill-climb μ (distance)"
if st.sidebar.button(hill_label):
    from persistence import get_dataset_for_prompt_or_session
    Xd, yd = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
    try:
        if Xd is not None and yd is not None and getattr(Xd, 'shape', (0,))[0] > 0:
            r_val = None if float(trust_mu) <= 0.0 else float(trust_mu)
            if vm_choice == "XGBoost":
                # XGB-guided multi-step hill climb: use Ridge for direction and XGB as critic.
                from value_scorer import get_value_scorer
                cache = st.session_state.get('xgb_cache') or {}
                mdl = cache.get('model')
                if mdl is not None:
                    scorer = get_value_scorer("XGBoost", lstate, base_prompt, st.session_state)
                    step_scale = float(lr_mu_ui) * float(getattr(lstate, "sigma", 1.0))
                    # Use Optimization steps (latent) numeric control for XGB hill steps.
                    try:
                        steps_now = int(st.session_state.get("iter_steps", iter_steps))
                    except Exception:
                        steps_now = int(iter_steps)
                    ll.hill_climb_mu_xgb(lstate, base_prompt, scorer, steps=steps_now, step_scale=step_scale, trust_r=r_val)
                    save_state(lstate, st.session_state.state_path)
            else:
                ll.hill_climb_mu_distance(lstate, base_prompt, Xd, yd, eta=float(eta_mu), gamma=float(gamma_mu), trust_r=r_val)
                save_state(lstate, st.session_state.state_path)
    except Exception:
        pass
    if callable(st_rerun):
        st_rerun()
# Value function option: controlled solely by the dropdown above
use_xgb = (vm_choice == "XGBoost")

# Legacy toggles (collapsed when dropdown exists) — kept to preserve tests
def _legacy_mode_controls():
    cm = st.sidebar.checkbox("Batch curation mode", value=False)
    aq = st.sidebar.checkbox("Async queue mode", value=False)
    return cm, aq

if selected_gen_mode is not None and callable(getattr(st.sidebar, 'expander', None)):
    with st.sidebar.expander("Advanced (legacy mode toggles)"):
        curation_mode_cb, async_queue_mode_cb = _legacy_mode_controls()
else:
    curation_mode_cb, async_queue_mode_cb = _legacy_mode_controls()

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
# Regularization λ: slider plus direct input for precise edits
reg_lambda = _sb_sld("Ridge λ (regularization)", 0.0, 1e5, value=1e-3)
# Allow precise manual entry; overrides slider when edited; default to 0.0 when unset
try:
    reg_lambda = float(_sb_num("Ridge λ (edit)", min_value=0.0, max_value=1e5, value=float(reg_lambda), step=1e-3, format="%.6f"))
except Exception:
    reg_lambda = float(reg_lambda)
try:
    st.session_state['reg_lambda'] = float(reg_lambda)
except Exception:
    pass
try:
    eta_default = float(getattr(st.session_state, "iter_eta", iter_eta))
except Exception:
    eta_default = float(iter_eta)
iter_eta_num = _sb_num("Iterative step (eta)", min_value=0.0, max_value=1.0, value=eta_default, step=0.01, format="%.2f")
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
    steps_default = int(getattr(st.session_state, "iter_steps", iter_steps))
except Exception:
    steps_default = int(iter_steps)
iter_steps_num = _sb_num("Optimization steps (latent)", min_value=0, max_value=10000, value=steps_default, step=1)
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

st.sidebar.subheader("State persistence")
_export_state_bytes = export_state_bytes  # back-compat for tests
render_persistence_controls(lstate, st.session_state.prompt, st.session_state.state_path, _apply_state, st_rerun)

from ui_metrics import render_iter_step_scores, render_mu_value_history
def _render_iter_step_scores():
    return render_iter_step_scores(st, lstate, base_prompt, vm_choice, int(iter_steps), float(iter_eta) if iter_eta is not None else None, trust_r)
_render_iter_step_scores()
render_mu_value_history(st, lstate, base_prompt)

# Autorun: set model once so decode paths are ready
set_model(selected_model)

# Latent state summary (concise)
st.sidebar.subheader("Latent state")
if not hasattr(st.sidebar, 'write'):
    # Fallback for test stubs that don't implement .write
    st.sidebar.text = getattr(st.sidebar, 'text', lambda *a, **k: None)
    def _w(x):
        st.sidebar.text(str(x))
    st.sidebar.write = _w
# Ensure st.metric exists in test stubs; map to write format if missing
if not hasattr(st.sidebar, 'metric'):
    st.sidebar.metric = lambda label, value, **k: st.sidebar.write(f"{label}: {value}")

# Data counters at the very top of the sidebar
try:
    _info_top = state_summary(lstate)
    st.sidebar.subheader("Interaction data (in-memory)")
    rows = [("Pairs", f"{_info_top['pairs_logged']}"), ("Choices", f"{_info_top['choices_logged']}")]
    # Simple train score over logged pairs using current w
    try:
        X = getattr(lstate, 'X', None)
        y = getattr(lstate, 'y', None)
        if X is not None and y is not None and len(y) > 0:
            pred = X @ lstate.w
            acc = float(np.mean((pred >= 0) == (y > 0)))
            rows.append(("Train score", f"{acc*100:.0f}%"))
        else:
            rows.append(("Train score", "n/a"))
    except Exception:
        rows.append(("Train score", "n/a"))
    # Dataset rows (from saved NPZ)
    try:
        rows.append(("Dataset rows", str(dataset_rows_for_prompt(base_prompt))))
    except Exception:
        rows.append(("Dataset rows", "n/a"))
    sidebar_metric_rows(rows, per_row=2)
except Exception:
    pass

# Training data details (collapsed)
try:
    # Explain how we build z and latents
    _exp2 = getattr(st.sidebar, 'expander', None)
    def _latent_creation_writes():
        import hashlib as _hl
        z_p = z_from_prompt(lstate, base_prompt)
        h = _hl.sha1(base_prompt.encode('utf-8')).hexdigest()[:10]
        st.sidebar.subheader("Latent creation details")
        st.sidebar.write(f"Prompt hash: {h}")
        st.sidebar.write("z_prompt = RNG(prompt_sha1) → N(0,1)^d · σ")
        st.sidebar.write("Batch sample: z = z_prompt + σ · 0.8 · r, r=unit Gaussian")
        st.sidebar.write(f"‖z_prompt‖ = {float(np.linalg.norm(z_p)):.3f}, σ = {float(lstate.sigma):.3f}")
        st.sidebar.write(f"Latents shape: (1,4,{lstate.height//8},{lstate.width//8}), noise_gamma=0.35, per‑channel zero‑mean")
    if callable(_exp2):
        with _exp2("Latent creation", expanded=False):
            _latent_creation_writes()
    else:
        _latent_creation_writes()
except Exception:
    pass

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
        # Fallback simple line
        st.sidebar.write("Training data: pos={} neg={} d={}".format(stats.get("pos",0), stats.get("neg",0), stats.get("d",0)))
except Exception:
    pass

# Value model details (collapsed)
try:
    exp = getattr(st.sidebar, 'expander', None)
    if callable(exp):
        with exp("Value model", expanded=False):
            # Model type and status
            vm = "Ridge"
            cache = st.session_state.get('xgb_cache') or {}
            if use_xgb and cache.get('model') is not None:
                vm = "XGBoost"
            st.sidebar.write(f"Value model: {vm}")
            # Value scorer status via helper
            try:
                from value_scorer import get_value_scorer_with_status
                _scorer_vm, scorer_status = get_value_scorer_with_status(vm_choice, lstate, base_prompt, st.session_state)
            except Exception:
                scorer_status = "unknown"
            st.sidebar.write(f"Value scorer status: {scorer_status}")
            if vm == "Ridge":
                try:
                    w_norm = float(np.linalg.norm(lstate.w))
                except Exception:
                    w_norm = 0.0
                rows = 0
                try:
                    rows = int(dataset_rows_for_prompt(base_prompt))
                except Exception:
                    rows = 0
                st.sidebar.write(f"λ={reg_lambda:.3g}, ||w||={w_norm:.3f}, rows={rows}")
            else:
                # Show cached rows used for last fit and basic params
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
                # When XGBoost is active and enough data is available, show
                # both XGB- and Ridge-based CV scores for transparency.
                try:
                    from persistence import get_dataset_for_prompt_or_session as _get_ds_vm
                    from metrics import ridge_cv_accuracy as _rcv_vm, xgb_cv_accuracy as _xcv_vm
                    Xv, yv = _get_ds_vm(base_prompt, st.session_state)
                    if Xv is not None and yv is not None and len(yv) >= 4:
                        import numpy as _np
                        n_rows_vm = int(len(yv))
                        # XGB-based CV (mirrors Data block behaviour, but uses a fixed small k for speed)
                        try:
                            k_xgb = min(3, n_rows_vm)
                            cv_xgb = float(_xcv_vm(Xv, yv, k_xgb))
                            if not _np.isnan(cv_xgb):
                                st.sidebar.write(f"CV (XGBoost): {cv_xgb*100:.0f}% (k={k_xgb})")
                        except Exception:
                            pass
                        # Ridge-based CV for comparison
                        try:
                            k_ridge = min(5, n_rows_vm)
                            cv_r = float(_rcv_vm(Xv, yv, lam=float(st.session_state.get('reg_lambda', 1e-3)), k=k_ridge))
                            if not _np.isnan(cv_r):
                                st.sidebar.write(f"CV (Ridge): {cv_r*100:.0f}% (k={k_ridge})")
                        except Exception:
                            pass
                except Exception:
                    pass
    else:
        st.sidebar.write("Value model: Ridge")
except Exception:
    pass

try:
    _exp = getattr(st.sidebar, 'expander', None)
    if callable(_exp):
        with _exp("Environment", expanded=False):
            env_panel(get_env_summary())
    else:
        env_panel(get_env_summary())
except Exception:
    env_panel(get_env_summary())

try:
    perf_panel(get_last_call() or {}, st.session_state.get('last_train_ms'))
except Exception:
    pass

info = state_summary(lstate)
pairs_state = [("Latent dim", f"{info['d']}")]
pairs_state += [(k, f"{info[k]}") for k in ('width','height','step','sigma','mu_norm','w_norm','pairs_logged','choices_logged')]
sidebar_metric_rows(pairs_state, per_row=2)

# Debug panel (collapsible): expose last pipeline call stats to spot black-frame issues
try:
    expander = getattr(st.sidebar, 'expander', None)
    if callable(expander):
        with expander("Debug", expanded=False):
            last = get_last_call() or {}
            dbg_pairs = []
            for k in ("model_id", "width", "height", "steps", "guidance", "latents_std", "init_sigma", "img0_std", "img0_min", "img0_max"):
                if k in last and last[k] is not None:
                    dbg_pairs.append((k, str(last[k])))
            if is_turbo:
                dbg_pairs.append(("guidance_eff", str(guidance_eff)))
            try:
                retry_val = os.getenv("RETRY_ON_OOM", "0")
                dbg_pairs.append(("RETRY_ON_OOM", retry_val))
                toggle = getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
                    "Enable OOM retry (env RETRY_ON_OOM)",
                    value=(retry_val not in ("0", "false", "False", "")),
                )
                os.environ["RETRY_ON_OOM"] = "1" if toggle else "0"
            except Exception:
                pass
            if dbg_pairs:
                sidebar_metric_rows(dbg_pairs, per_row=2)
            try:
                ls = last.get('latents_std')
                if ls is not None and float(ls) <= 1e-6:
                    st.sidebar.write('warn: latents std ~0')
            except Exception:
                pass
    else:
        # Fallback when expander not available (e.g., test stubs)
        st.sidebar.subheader("Debug")
        last = get_last_call() or {}
        dbg_pairs = []
        for k in ("model_id", "width", "height", "steps", "guidance", "latents_std", "init_sigma", "img0_std", "img0_min", "img0_max"):
            if k in last and last[k] is not None:
                dbg_pairs.append((k, str(last[k])))
        if is_turbo:
            dbg_pairs.append(("guidance_eff", str(guidance_eff)))
        try:
            retry_val = os.getenv("RETRY_ON_OOM", "0")
            dbg_pairs.append(("RETRY_ON_OOM", retry_val))
            toggle = getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
                "Enable OOM retry (env RETRY_ON_OOM)",
                value=(retry_val not in ("0", "false", "False", "")),
            )
            os.environ["RETRY_ON_OOM"] = "1" if toggle else "0"
        except Exception:
            pass
        if dbg_pairs:
            sidebar_metric_rows(dbg_pairs, per_row=2)
        try:
            ls = last.get('latents_std')
            if ls is not None and float(ls) <= 1e-6:
                st.sidebar.write('warn: latents std ~0')
        except Exception:
            pass
except Exception:
    pass

# State metadata panel and file paths
render_metadata_panel(st.session_state.state_path, st.session_state.prompt)
# Paths/Dataset viewer and Manage states panels removed to declutter sidebar

status_panel(st.session_state.images, st.session_state.mu_image)

# (legacy Debug checkbox block removed; unified Debug expander exists above)

from pair_ui import generate_pair as _pair_generate, _prefetch_next_for_generate as _pair_prefetch, _pair_scores as _pair_scores_impl


def generate_pair():
    try:
        print("[pair] generate_pair() called")
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


def _curation_sample_one() -> np.ndarray:
    return _batch_ui._sample_around_prompt(0.8)


def _sample_around_prompt(scale: float = 0.8) -> np.ndarray:
    return _batch_ui._sample_around_prompt(scale)


def _curation_replace_at(idx: int) -> None:
    return _batch_ui._curation_replace_at(idx)


def _proposer_opts():
    """Return a ProposerOpts built from current sidebar settings (delegates)."""
    from proposer import build_proposer_opts
    try:
        ie = float(iter_eta) if iter_eta is not None else None
    except Exception:
        ie = None
    return build_proposer_opts(int(iter_steps), ie, trust_r, gamma_orth)


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
            fit_value_model(st.session_state.get('vm_train_choice', st.session_state.get('vm_choice')),
                           lstate, X, y, lam_now, _st.session_state)
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


def _choose_preference(side: str) -> None:
    """Handle pair preference: update state, persist both labels, propose next pair."""
    import streamlit as _st
    set_model(selected_model)
    z_p = z_from_prompt(lstate, base_prompt)
    feats_a = z_a - z_p
    feats_b = z_b - z_p
    winner = 'a' if side == 'a' else 'b'
    lam_now = float(getattr(_st.session_state, 'reg_lambda', reg_lambda))
    update_latent_ridge(lstate, z_a, z_b, winner, lr_mu=float(lr_mu_ui), lam=lam_now, feats_a=feats_a, feats_b=feats_b)
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
        print("[mode] running Batch curation")
    except Exception:
        pass
    return _batch_ui.run_batch_mode()


def run_upload_mode() -> None:
    st.subheader("Upload latents")
    lstate, prompt = _batch_ui._lstate_and_prompt()
    uploads = getattr(st.sidebar, "file_uploader", lambda *a, **k: [])(
        "Upload images to use as latents", accept_multiple_files=True, type=["png", "jpg", "jpeg", "webp"]
    )
    steps = int(getattr(st.session_state, "steps", 6))
    guidance_eff = float(getattr(st.session_state, "guidance_eff", 0.0))
    z_p = z_from_prompt(lstate, prompt)
    nonce = int(st.session_state.get("cur_batch_nonce", 0))
    if not uploads:
        st.write("Upload at least one image to score it as Good/Bad.")
        return
    for idx, upl in enumerate(uploads):
        try:
            img_raw = Image.open(upl)
        except Exception:
            continue
        try:
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
            up_dir = os.path.join("data", h, "uploads")
            os.makedirs(up_dir, exist_ok=True)
            fname = f"upload_{nonce}_{idx}.png"
            img_raw.save(os.path.join(up_dir, fname))
        except Exception:
            pass
        z = image_to_z(img_raw, lstate)
        try:
            lat = z.reshape(1, 4, lstate.height // 8, lstate.width // 8)
        except Exception:
            continue
        img_dec = generate_flux_image_latents(
            prompt,
            latents=lat,
            width=lstate.width,
            height=lstate.height,
            steps=steps,
            guidance=guidance_eff,
        )
        st.image(img_dec, caption=f"Upload {idx}", width="stretch")
        w_slider = st.slider(
            f"Weight upload {idx}",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key=f"upl_w_{nonce}_{idx}",
        )
        if st.button(f"Good (+1) upload {idx}", key=f"upl_good_{nonce}_{idx}"):
            _curation_add(float(w_slider), z, img_dec)
            _curation_train_and_next()
        if st.button(f"Bad (-1) upload {idx}", key=f"upl_bad_{nonce}_{idx}"):
            _curation_add(-float(w_slider), z, img_dec)
            _curation_train_and_next()


from modes import run_mode
# Run selected mode (Batch default vs Async queue)
try:
    async_queue_mode
except NameError:  # minimal guard for test stubs/import order
    async_queue_mode = False
try:
    print(f"[mode] dispatch async_queue_mode={bool(async_queue_mode)}")
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
