import streamlit as st
import numpy as np
from typing import Optional, Tuple, Any
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from constants import (
    DEFAULT_PROMPT,
    DEFAULT_MODEL,
    MODEL_CHOICES,
)
from constants import Config
from env_info import get_env_summary
from ui import sidebar_metric_rows, render_pair_sidebar, env_panel, status_panel
from persistence import state_path_for_prompt, export_state_bytes, dataset_path_for_prompt, dataset_rows_for_prompt, append_dataset_row, dataset_stats_for_prompt
import background as bg
from persistence_ui import render_persistence_controls, render_metadata_panel, render_paths_panel, render_dataset_viewer
from latent_opt import (
    init_latent_state,
    propose_latent_pair_ridge,
    propose_next_pair,
    z_to_latents,
    z_from_prompt,
    update_latent_ridge,
    save_state,
    load_state,
    state_summary,
)
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

st.set_page_config(page_title="Latent Preference Optimizer", layout="centered")
# Streamlit rerun API shim: prefer st.rerun(), fallback to experimental in older versions
st_rerun = getattr(st, 'rerun', getattr(st, 'experimental_rerun', None))
st.title("Optimize Latents by Preference — FLUX (Local GPU)")
st.caption("Fixed prompt; we optimize the latent vector directly from your choices.")

# Back-compat for tests: keep names on app module
_state_path_for_prompt = state_path_for_prompt

# Prompt-aware persistence
if 'prompt' not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT

base_prompt = st.text_input("Prompt", value=st.session_state.prompt)
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
    st.image(img, caption=caption, use_container_width=True)
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
st.sidebar.subheader("Mode")
_gen_opts = ["Batch curation", "Async queue"]
selected_gen_mode = None
if callable(_sb_sel):
    try:
        selected_gen_mode = _sb_sel("Generation mode", _gen_opts, index=0)
        if selected_gen_mode not in _gen_opts:
            selected_gen_mode = None
    except Exception:
        selected_gen_mode = None

_vm_opts = ["DistanceHill", "Ridge", "XGBoost", "CosineHill"]
vm_choice = None
if callable(_sb_sel):
    try:
        vm_choice = _sb_sel("Value model", _vm_opts, index=0)
        if vm_choice not in _vm_opts:
            vm_choice = "DistanceHill"
    except Exception:
        vm_choice = "DistanceHill"
else:
    vm_choice = "DistanceHill"
st.session_state['vm_choice'] = vm_choice

# Pair proposer dropdown removed; proposer is derived from Value model.

# Quick data strip
try:
    st.sidebar.subheader("Data")
    # Dataset rows from persisted NPZ
    try:
        _rows_cnt = int(dataset_rows_for_prompt(base_prompt))
    except Exception:
        _rows_cnt = 0
    # Train score using selected value model (Ridge or XGBoost)
    try:
        X_ = getattr(lstate, 'X', None)
        y_ = getattr(lstate, 'y', None)
        _train_score = "n/a"
        if X_ is not None and y_ is not None and len(y_) > 0:
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
    # Minimal K-fold CV over in-memory X/y or n/a
    try:
        from metrics import ridge_cv_accuracy as _rcv
        if X_ is not None and y_ is not None and len(y_) >= 4:
            _k = min(5, int(len(y_)))
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
            _vm_settings = "n=50,depth=3"
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
    sidebar_metric_rows([("CV score", _cv_score), ("Last train", _last_train)], per_row=2)
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
    except Exception:
        pass
except Exception:
    pass
width = _sb_num("Width", min_value=256, max_value=1024, step=64, value=lstate.width)
height = _sb_num("Height", min_value=256, max_value=1024, step=64, value=lstate.height)
steps = _sb_sld("Steps", 1, 50, value=Config.DEFAULT_STEPS)
guidance = _sb_sld("Guidance", 0.0, 10.0, value=Config.DEFAULT_GUIDANCE, step=0.1)
if st.sidebar.button("Apply size now"):
    _apply_state(init_latent_state(width=int(width), height=int(height)))
    save_state(st.session_state.lstate, st.session_state.state_path)
    _toast(f"Applied size {int(width)}x{int(height)}")
    if callable(st_rerun):
        st_rerun()
st.sidebar.header("Settings")
"""Model selection"""
_model_sel = getattr(st.sidebar, 'selectbox', None)
if callable(_model_sel):
    try:
        selected_model = _model_sel("Model", MODEL_CHOICES, index=0)
    except Exception:
        selected_model = DEFAULT_MODEL
else:
    selected_model = DEFAULT_MODEL

## Generation mode moved to top-of-sidebar (see "Mode" section above)
# Simplified: hardcode sd-turbo; no model selector
_exp = getattr(st.sidebar, 'expander', None)
if callable(_exp):
    with _exp("Proposer controls", expanded=False):
        # Distance hill climbing settings: use numeric inputs for precision
        alpha = _sb_num("Alpha (ridge d1)", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f")
        beta = _sb_num("Beta (ridge d2)", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f")
        trust_r = _sb_num("Trust radius (||y||)", min_value=0.0, max_value=5.0, value=2.5, step=0.1, format="%.1f")
        lr_mu_ui = _sb_num("Step size (lr_μ)", min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.2f")
        gamma_orth = _sb_num("Orth explore (γ)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.2f")
        # Optional iterative controls (default disabled)
        iter_steps = _sb_num("Optimization steps (latent)", min_value=1, max_value=10, value=1, step=1)
        try:
            # Show concise iteration summary and per-step predicted values
            st.sidebar.write(f"Iter steps: {int(iter_steps)}")
            def _iter_vals(n_steps: int, trust_r_v, sigma_v, eta_v, w_now) -> list[str]:
                wn = float(np.linalg.norm(w_now))
                n_steps = max(1, int(n_steps))
                if eta_v is not None and float(eta_v) > 0.0:
                    step_sz = float(eta_v)
                elif trust_r_v is not None and float(trust_r_v) > 0.0:
                    step_sz = float(trust_r_v) / n_steps
                else:
                    step_sz = float(sigma_v) / n_steps
                return [f"{(k * step_sz * wn):.3f}" for k in range(1, n_steps + 1)]
            st.sidebar.write("Step values (pred.): " + ", ".join(_iter_vals(iter_steps, trust_r, lstate.sigma, None, lstate.w)[:6]))
        except Exception:
            pass
else:
    # Fallback path for minimal stubs: number inputs without hard caps
    alpha = _sb_num("Alpha (ridge d1)", min_value=0.0, value=0.5, step=0.01)
    beta = _sb_num("Beta (ridge d2)", min_value=0.0, value=0.5, step=0.01)
    trust_r = _sb_num("Trust radius (||y||)", min_value=0.0, value=2.5, step=0.1)
    lr_mu_ui = _sb_num("Step size (lr_μ)", min_value=0.0, value=0.3, step=0.01)
    gamma_orth = _sb_num("Orth explore (γ)", min_value=0.0, value=0.2, step=0.01)
    iter_steps = _sb_num("Optimization steps (latent)", min_value=1, value=10, step=1)
    try:
        st.sidebar.write(f"Iter steps: {int(iter_steps)}")
        def _iter_vals2(n_steps: int, trust_r_v, sigma_v, eta_v, w_now) -> list[str]:
            wn = float(np.linalg.norm(w_now))
            n_steps = max(1, int(n_steps))
            if eta_v is not None and float(eta_v) > 0.0:
                step_sz = float(eta_v)
            elif trust_r_v is not None and float(trust_r_v) > 0.0:
                step_sz = float(trust_r_v) / n_steps
            else:
                step_sz = float(sigma_v) / n_steps
            return [f"{(k * step_sz * wn):.3f}" for k in range(1, n_steps + 1)]
        st.sidebar.write("Step values (pred.): " + ", ".join(_iter_vals2(iter_steps, trust_r, lstate.sigma, None, lstate.w)[:6]))
    except Exception:
        pass
    # Minimal hill-climb μ (distance) controls
    st.sidebar.subheader("Hill-climb μ")
    eta_mu = _sb_num("η (step)", min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f")
    gamma_mu = _sb_num("γ (sigmoid)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, format="%.1f")
    trust_mu = _sb_num("Trust radius r (0=off)", min_value=0.0, max_value=1e3, value=0.0, step=1.0, format="%.1f")
    if st.sidebar.button("Hill-climb μ (distance)"):
        try:
            with np.load(dataset_path_for_prompt(base_prompt)) as d:
                Xd = d['X'] if 'X' in d.files else None
                yd = d['y'] if 'y' in d.files else None
        except Exception:
            Xd = yd = None
        if (Xd is None or yd is None) and getattr(st.session_state, 'dataset_X', None) is not None:
            Xd = st.session_state.dataset_X
            yd = st.session_state.dataset_y
        try:
            if Xd is not None and yd is not None and getattr(Xd, 'shape', (0,))[0] > 0:
                r_val = None if float(trust_mu) <= 0.0 else float(trust_mu)
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

_exp = getattr(st.sidebar, 'expander', None)
if callable(_exp):
    with _exp("Batch controls", expanded=(selected_gen_mode==_gen_opts[0])):
        batch_size = _sb_sld("Batch size", 2, 12, value=6, step=1)
    with _exp("Queue controls", expanded=(selected_gen_mode==_gen_opts[1])):
        queue_size = _sb_sld("Queue size", 2, 16, value=6, step=1)
else:
    batch_size = _sb_sld("Batch size", 2, 12, value=6, step=1)
    queue_size = _sb_sld("Queue size", 2, 16, value=6, step=1)

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
iter_eta = _sb_num("Iterative step (eta)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f")
use_clip = False

# 7 GB VRAM recipe: lighter model, smaller size, no CLIP
# 7 GB VRAM mode removed; users can lower size/steps manually via controls

is_turbo = True
guidance_eff = 0.0

# (auto-run added after function definitions below)

mu_show = False

st.sidebar.subheader("Persistence")
_export_state_bytes = export_state_bytes  # back-compat for tests
render_persistence_controls(lstate, st.session_state.prompt, st.session_state.state_path, _apply_state, st_rerun)

# Show latent optimization score per iterative step in the sidebar
def _render_iter_step_scores():
    try:
        # Load dataset (disk or from in-memory session)
        from persistence import get_dataset_for_prompt_or_session
        Xd, yd = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
        if Xd is None or yd is None or len(getattr(yd, 'shape', (0,))) == 0:
            return
        # Direction along current w
        w = getattr(lstate, 'w', None)
        if w is None:
            return
        w = w[: lstate.d]
        n = float(np.linalg.norm(w))
        if n == 0.0:
            return
        d1 = w / n
        # Step length policy
        n_steps = max(1, int(iter_steps))
        if iter_eta and float(iter_eta) > 0.0:
            step_len = float(iter_eta)
        elif trust_r and float(trust_r) > 0.0:
            step_len = float(trust_r) / n_steps
        else:
            step_len = float(lstate.sigma) / n_steps
        from latent_logic import z_from_prompt, distancehill_score, cosinehill_score
        z_p = z_from_prompt(lstate, base_prompt)
        scores = []
        for k in range(1, n_steps + 1):
            zc = z_p + (k * step_len) * d1
            if vm_choice == 'CosineHill':
                s = float(cosinehill_score(base_prompt, zc, lstate, Xd, yd, beta=5.0))
            elif vm_choice == 'DistanceHill':
                s = float(distancehill_score(base_prompt, zc, lstate, Xd, yd, gamma=0.5))
            elif vm_choice == 'XGBoost':
                try:
                    from xgb_value import score_xgb_proba
                    s = float(score_xgb_proba((st.session_state.get('xgb_cache') or {}).get('model'), (zc - z_p)))
                except Exception:
                    s = 0.0
            else:
                s = float(np.dot(lstate.w[: lstate.d], (zc - z_p)))
            scores.append(s)
        try:
            st.sidebar.write("Step scores: " + ", ".join(f"{v:.3f}" for v in scores[:8]))
        except Exception:
            pass
        try:
            from ui import sidebar_metric_rows
            pairs = []
            for i, v in enumerate(scores[:4], 1):
                pairs.append((f"Step {i}", f"{v:.3f}"))
            if pairs:
                sidebar_metric_rows(pairs, per_row=2)
        except Exception:
            pass
    except Exception:
        pass

_render_iter_step_scores()

# Autorun: generate prompt image only when missing or prompt changed; then generate the A/B pair
set_model(selected_model)
if st.session_state.get('prompt_image') is None or prompt_changed:
    if callable(generate_flux_image):
        st.session_state.prompt_image = generate_flux_image(
            base_prompt, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff
        )
    else:
        # Minimal fallback: decode prompt-derived latent via latents path
        z_prompt = z_from_prompt(lstate, base_prompt)
        lat = z_to_latents(lstate, z_prompt)
        st.session_state.prompt_image = generate_flux_image_latents(
            base_prompt, latents=lat, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff
        )
    try:
        st.session_state.prompt_stats = get_last_call().copy()
    except Exception:
        st.session_state.prompt_stats = {}


# Prompt-only generation
st.subheader("Prompt-only generation")
if st.button("Generate from Prompt", use_container_width=True):
    set_model(selected_model)
    if callable(generate_flux_image):
        st.session_state.prompt_image = generate_flux_image(
            base_prompt, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff
        )
    else:
        z_prompt = z_from_prompt(lstate, base_prompt)
        lat = z_to_latents(lstate, z_prompt)
        st.session_state.prompt_image = generate_flux_image_latents(
            base_prompt, latents=lat, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff
        )
    try:
        st.session_state.prompt_stats = get_last_call().copy()
    except Exception:
        st.session_state.prompt_stats = {}
if st.session_state.get('prompt_image') is not None:
    # Also show the prompt-derived latent summary for transparency
    try:
        z_prompt = z_from_prompt(st.session_state.lstate, base_prompt)
        st.caption(f"z_prompt: first8={np.array2string(z_prompt[:8], precision=2, separator=', ')} | ‖z_p‖={float(np.linalg.norm(z_prompt)):.3f}")
    except Exception:
        pass
    _image_fragment(st.session_state.prompt_image, caption="Prompt-only image")
    # Content warning badge for prompt image (if stats present)
    try:
        ps = st.session_state.get('prompt_stats') or {}
        std = float(ps.get('img0_std')) if 'img0_std' in ps and ps['img0_std'] is not None else None
        if std is not None and std < 2.0:
            warn = getattr(st, 'warning', None)
            (warn or st.write)(f"⚠️ Low content (std={std:.2f}). Try smaller size, steps 6, guidance 0.0 for Turbo.")
    except Exception:
        pass

# State info (concise)
st.sidebar.subheader("State info")
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
    st.sidebar.subheader("Data")
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
        st.sidebar.subheader("How latents are created")
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
            # Model type
            vm = "Ridge"
            cache = st.session_state.get('xgb_cache') or {}
            if use_xgb and cache.get('model') is not None:
                vm = "XGBoost"
            st.sidebar.write(f"Value model: {vm}")
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
                st.sidebar.write(f"fit_rows={int(n_fit)}, n_estimators=50, depth=3")
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

# Performance panel: show last decode and last training times
try:
    perf_exp = getattr(st.sidebar, 'expander', None)
    last = get_last_call() or {}
    dur_s = last.get('dur_s')
    train_ms = st.session_state.get('last_train_ms')
    pairs = []
    if dur_s is not None:
        pairs.append(("decode_s", f"{float(dur_s):.3f}"))
    if train_ms is not None:
        pairs.append(("train_ms", f"{float(train_ms):.1f}"))
    if callable(perf_exp):
        with perf_exp("Performance", expanded=False):
            if pairs:
                sidebar_metric_rows(pairs, per_row=2)
    else:
        if pairs:
            sidebar_metric_rows(pairs, per_row=2)
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
render_paths_panel(st.session_state.state_path, st.session_state.prompt)
render_dataset_viewer()

status_panel(st.session_state.images, st.session_state.mu_image)

# Vector info for current pair (render early so tests see it on import)
try:
        za, zb = st.session_state.lz_pair
        value_scorer = None
        if use_xgb:
            try:
                from xgb_value import fit_xgb_classifier, score_xgb_proba  # type: ignore
                # Cache model in session; retrain only when sample count changes
                X = getattr(lstate, 'X', None)
                y = getattr(lstate, 'y', None)
                n = 0 if (y is None) else len(y)
                cache = st.session_state.get('xgb_cache') or {}
                mdl, last_n = cache.get('model'), cache.get('n')
                if X is not None and y is not None and n > 0 and len(set(y.tolist())) > 1:
                    if mdl is None or last_n != n:
                        mdl = fit_xgb_classifier(X, y)
                        st.session_state.xgb_cache = {'model': mdl, 'n': n}
                        try:
                            from datetime import datetime, timezone
                            st.session_state['last_train_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
                        except Exception:
                            pass
                    value_scorer = lambda f: score_xgb_proba(mdl, f)
            except Exception:
                value_scorer = None
        if st.session_state.get('vm_choice') == 'DistanceHill':
            # Build dataset from disk or in-memory for distance scoring
            from persistence import get_dataset_for_prompt_or_session
            Xd, yd = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
            if Xd is not None and yd is not None and getattr(Xd, 'shape', (0,))[0] > 0:
                from latent_logic import distancehill_score
                z_p_here = z_from_prompt(lstate, base_prompt)
                def _score_distance(fvec):
                    zc = z_p_here + np.asarray(fvec, dtype=float)
                    return float(distancehill_score(base_prompt, zc, lstate, Xd, yd, gamma=0.5))
                value_scorer = _score_distance
        if st.session_state.get('vm_choice') == 'CosineHill':
            # Build dataset for cosine scoring
            from persistence import get_dataset_for_prompt_or_session
            Xd, yd = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
            if Xd is not None and yd is not None and getattr(Xd, 'shape', (0,))[0] > 0:
                from latent_logic import cosinehill_score
                z_p_here = z_from_prompt(lstate, base_prompt)
                def _score_cos(fvec):
                    zc = z_p_here + np.asarray(fvec, dtype=float)
                    return float(cosinehill_score(base_prompt, zc, lstate, Xd, yd, beta=5.0))
                value_scorer = _score_cos
        render_pair_sidebar(st.session_state.lstate, st.session_state.prompt, za, zb, lr_mu_val=float(lr_mu_ui), value_scorer=value_scorer)
except Exception:
    pass

# Manage states (per‑prompt)
st.sidebar.subheader("Manage states")
recent = st.session_state.get('recent_prompts', [])
# Maintain MRU list with current prompt at front
if not recent or recent[0] != st.session_state.prompt:
    recent = [st.session_state.prompt] + [p for p in recent if p != st.session_state.prompt]
st.session_state.recent_prompts = recent[:5]
if len(st.session_state.recent_prompts) > 1:
    opts = st.session_state.recent_prompts
    sel = st.sidebar.selectbox("Recent prompts", opts, index=0)
    if sel not in opts:
        sel = opts[0]
    if st.sidebar.button("Switch prompt") and sel != st.session_state.prompt:
        st.session_state.prompt = sel
        st.session_state.state_path = state_path_for_prompt(sel)
        if os.path.exists(st.session_state.state_path):
            _apply_state(load_state(st.session_state.state_path))
        else:
            _apply_state(init_latent_state())
        if callable(st_rerun):
            st_rerun()

# (legacy Debug checkbox block removed; unified Debug expander exists above)

def _decode_one(side: str, latents: np.ndarray) -> Any:
    """Decode one side and record last-call stats (no UI rendering here)."""
    img = generate_flux_image_latents(base_prompt, latents=latents, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff)
    try:
        st.session_state.img_stats = st.session_state.get('img_stats') or {}
        st.session_state.img_stats[side] = get_last_call().copy()
    except Exception:
        pass
    return img


def generate_pair():
    lat_a = z_to_latents(lstate, z_a)
    lat_b = z_to_latents(lstate, z_b)
    img_a = _decode_one('left', lat_a)
    img_b = _decode_one('right', lat_b)
    st.session_state.images = (img_a, img_b)
    # Prefetch next pair for the Generate button
    _prefetch_next_for_generate()


def _prefetch_next_for_generate():
    try:
        try:
            if st.session_state.get('vm_choice') == 'DistanceHill':
                # Build dataset
                try:
                    with np.load(dataset_path_for_prompt(base_prompt)) as d:
                        Xd = d['X'] if 'X' in d.files else None
                        yd = d['y'] if 'y' in d.files else None
                except Exception:
                    Xd = yd = None
                if (Xd is None or yd is None) and getattr(st.session_state, 'dataset_X', None) is not None:
                    Xd = st.session_state.dataset_X
                    yd = st.session_state.dataset_y
                from latent_logic import propose_pair_distancehill
                za_n, zb_n = propose_pair_distancehill(lstate, base_prompt, Xd, yd, alpha=float(alpha), gamma=0.5, trust_r=None)
            elif st.session_state.get('vm_choice') == 'CosineHill':
                try:
                    with np.load(dataset_path_for_prompt(base_prompt)) as d:
                        Xd = d['X'] if 'X' in d.files else None
                        yd = d['y'] if 'y' in d.files else None
                except Exception:
                    Xd = yd = None
                if (Xd is None or yd is None) and getattr(st.session_state, 'dataset_X', None) is not None:
                    Xd = st.session_state.dataset_X
                    yd = st.session_state.dataset_y
                from latent_logic import propose_pair_cosinehill
                za_n, zb_n = propose_pair_cosinehill(lstate, base_prompt, Xd, yd, alpha=float(alpha), beta=5.0, trust_r=None)
            else:
                za_n, zb_n = propose_next_pair(lstate, base_prompt, opts=_proposer_opts())
        except Exception:
            za_n, zb_n = propose_latent_pair_ridge(lstate)
        la_n = z_to_latents(lstate, za_n)
        lb_n = z_to_latents(lstate, zb_n)
        f = bg.schedule_decode_pair(base_prompt, la_n, lb_n, lstate.width, lstate.height, steps, guidance_eff)
        st.session_state.next_prefetch = {
            'za': za_n,
            'zb': zb_n,
            'f': f,
        }
    except Exception:
        st.session_state.pop('next_prefetch', None)

    # μ preview removed


# history helpers removed

def _curation_init_batch() -> None:
    if st.session_state.get('cur_batch') is None:
        st.session_state.cur_batch = []
        st.session_state.cur_labels = []
        _curation_new_batch()


def _curation_new_batch() -> None:
    z_list = []
    z_p = z_from_prompt(lstate, base_prompt)
    for i in range(int(batch_size)):
        # sample a small random delta around prompt
        r = lstate.rng.standard_normal(lstate.d)
        r = r / (np.linalg.norm(r) + 1e-12)
        z = z_p + lstate.sigma * 0.8 * r
        z_list.append(z)
    st.session_state.cur_batch = z_list
    st.session_state.cur_labels = [None] * len(z_list)
    # Reset per-item decode futures for async rendering
    st.session_state.batch_futures = [None] * len(z_list)
    st.session_state.batch_started = [None] * len(z_list)


def _curation_sample_one() -> np.ndarray:
    return _sample_around_prompt(0.8)


def _sample_around_prompt(scale: float = 0.8) -> np.ndarray:
    z_p = z_from_prompt(lstate, base_prompt)
    r = lstate.rng.standard_normal(lstate.d)
    r = r / (np.linalg.norm(r) + 1e-12)
    return z_p + lstate.sigma * float(scale) * r


def _curation_replace_at(idx: int) -> None:
    try:
        z_new = _curation_sample_one()
        st.session_state.cur_batch[idx] = z_new
        st.session_state.cur_labels[idx] = None
        _toast(f"Replaced item {idx}")
    except Exception:
        pass


def _proposer_opts():
    """Return a ProposerOpts built from current sidebar settings."""
    from latent_opt import ProposerOpts  # local import keeps tests light
    mode = 'iter' if (iter_steps > 1 or iter_eta > 0.0) else 'line'
    eta = float(iter_eta) if iter_eta > 0.0 else None
    return ProposerOpts(mode=mode, trust_r=trust_r, gamma=gamma_orth, steps=int(iter_steps), eta=eta)


def _curation_add(label: int, z: np.ndarray) -> None:
    # Feature is delta to prompt
    z_p = z_from_prompt(lstate, base_prompt)
    X = getattr(st.session_state, 'dataset_X', None)
    y = getattr(st.session_state, 'dataset_y', None)
    feat = (z - z_p).reshape(1, -1)
    lab = np.array([float(label)])
    st.session_state.dataset_X = feat if X is None else np.vstack([X, feat])
    st.session_state.dataset_y = lab if y is None else np.concatenate([y, lab])
    # Persist to on-disk dataset immediately (always train from saved dataset)
    try:
        append_dataset_row(base_prompt, feat, float(label))
        _toast(f"Saved label {int(label):+d} to dataset")
    except Exception:
        pass


def _curation_train_and_next() -> None:
    # Always train from saved dataset on disk; measure performance
    import streamlit as _st
    import time as _time
    from persistence import get_dataset_for_prompt_or_session
    X, y = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
    if X is not None and y is not None and getattr(X, 'shape', (0,))[0] > 0:
        try:
            lam_now = float(getattr(_st.session_state, 'reg_lambda', reg_lambda))
            t0 = _time.perf_counter()
            lstate.w = ll.ridge_fit(X, y, lam=lam_now)
            from datetime import datetime, timezone
            _st.session_state['last_train_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
            dt_ms = (_time.perf_counter() - t0) * 1000.0
            _st.session_state['last_train_ms'] = float(dt_ms)
            try:
                print(f"[perf] ridge fit: rows={X.shape[0]} d={X.shape[1]} took {dt_ms:.1f} ms")
            except Exception:
                pass
        except Exception:
            pass
    _curation_new_batch()


def _refit_from_dataset_keep_batch() -> None:
    """Refit ridge from saved dataset (or in-memory) without regenerating the batch."""
    import streamlit as _st
    from persistence import get_dataset_for_prompt_or_session
    X, y = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
    try:
        if X is not None and y is not None and getattr(X, 'shape', (0,))[0] > 0:
            lam_now = float(getattr(_st.session_state, 'reg_lambda', reg_lambda))
            lstate.w = ll.ridge_fit(X, y, lam=lam_now)
            from datetime import datetime, timezone
            _st.session_state['last_train_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
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


def _render_pair_ui(img_left: Any, img_right: Any,
                    d_left: Optional[float], d_right: Optional[float],
                    v_left: Optional[float], v_right: Optional[float]) -> None:
    left, right = st.columns(2)
    with left:
        if img_left is not None:
            cap = f"Left (d_prompt={d_left:.3f})" if d_left is not None else "Left"
            _image_fragment(img_left, caption=cap, v_label="V(left)", v_val=v_left)
        if st.button("Prefer Left", use_container_width=True):
            _choose_preference('a')
    with right:
        if img_right is not None:
            cap = f"Right (d_prompt={d_right:.3f})" if d_right is not None else "Right"
            _image_fragment(img_right, caption=cap, v_label="V(right)", v_val=v_right)
        if st.button("Prefer Right", use_container_width=True):
            _choose_preference('b')


def _render_batch_ui() -> None:
    st.subheader("Curation batch")
    futs = st.session_state.get('batch_futures') or []
    if not futs or len(futs) != len(st.session_state.cur_batch or []):
        futs = [None] * len(st.session_state.cur_batch or [])
        st.session_state.batch_futures = futs
    starts = st.session_state.get('batch_started') or [None] * len(st.session_state.cur_batch or [])
    if len(starts) != len(st.session_state.cur_batch or []):
        starts = [None] * len(st.session_state.cur_batch or [])
        st.session_state.batch_started = starts
    for i, z_i in enumerate(st.session_state.cur_batch or []):
        # Schedule decode if needed
        if futs[i] is None:
            la = z_to_latents(lstate, z_i)
            futs[i] = bg.schedule_decode_latents(base_prompt, la, lstate.width, lstate.height, steps, guidance_eff)
            st.session_state.batch_futures = futs
            import time as _time
            starts[i] = _time.time()
            st.session_state.batch_started = starts
        # If pending too long, synchronously decode to avoid indefinite loading
        def _sync_decode():
            la2 = z_to_latents(lstate, z_i)
            return generate_flux_image_latents(base_prompt, latents=la2, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff)
        img_i, futs[i] = bg.result_or_sync_after(futs[i], starts[i], 3.0, _sync_decode)
        st.session_state.batch_futures = futs
        if img_i is not None:
            _image_fragment(img_i, caption=f"Item {i}")
        else:
            st.write(f"Item {i}: loading…")
            # Do not render action buttons until the image is ready
            continue
        if st.button(f"Good (+1) {i}", use_container_width=True):
            import time as _time
            t0 = _time.perf_counter()
            _curation_add(1, z_i)
            st.session_state.cur_labels[i] = 1
            _refit_from_dataset_keep_batch()
            # Replace only this position to avoid full-batch stalls
            _curation_replace_at(i)
            try:
                print(f"[perf] good_label item={i} took {(_time.perf_counter()-t0)*1000:.1f} ms")
            except Exception:
                pass
            if callable(st_rerun):
                st_rerun()
        if st.button(f"Bad (-1) {i}", use_container_width=True):
            import time as _time
            t0 = _time.perf_counter()
            _curation_add(-1, z_i)
            st.session_state.cur_labels[i] = -1
            _refit_from_dataset_keep_batch()
            # Replace only this position to avoid full-batch stalls
            _curation_replace_at(i)
            try:
                print(f"[perf] bad_label item={i} took {(_time.perf_counter()-t0)*1000:.1f} ms")
            except Exception:
                pass
            if callable(st_rerun):
                st_rerun()
    if st.button("Train on dataset and next batch", type="primary"):
        _curation_train_and_next()
        if callable(st_rerun):
            st_rerun()
    if st.button("Train on dataset (keep batch)"):
        # Refit w from saved dataset without regenerating the batch
        _refit_from_dataset_keep_batch()
        if callable(st_rerun):
            st_rerun()


def _render_queue_ui() -> None:
    st.subheader("Async queue")
    q = st.session_state.get('queue') or []
    # Only show a single head-of-queue item in the UI
    if not q:
        st.write("Queue empty…")
    else:
        i = 0
        it = q[0]
        img = it['future'].result() if it['future'].done() else None
        if img is not None:
            _image_fragment(img, caption=f"Item {i}")
        else:
            st.write(f"Item {i}: loading…")
        if st.button(f"Accept {i}", use_container_width=True):
            _queue_label(i, 1)
            if callable(st_rerun):
                st_rerun()
        if st.button(f"Reject {i}", use_container_width=True):
            _queue_label(i, -1)
            if callable(st_rerun):
                st_rerun()
    _queue_fill_up_to()


def _pair_scores() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute d_left, d_right, V(left), V(right) for current pair."""
    try:
        z_p = z_from_prompt(st.session_state.lstate, base_prompt)
        d_left = float(np.linalg.norm(z_a - z_p))
        d_right = float(np.linalg.norm(z_b - z_p))
        try:
            if use_xgb:
                cache = st.session_state.get('xgb_cache') or {}
                mdl = cache.get('model')
                if mdl is not None:
                    from xgb_value import score_xgb_proba  # type: ignore
                    v_left = score_xgb_proba(mdl, (z_a - z_p))
                    v_right = score_xgb_proba(mdl, (z_b - z_p))
                else:
                    v_left = v_right = None
            else:
                w_now = st.session_state.lstate.w
                v_left = float(np.dot(w_now, (z_a - z_p)))
                v_right = float(np.dot(w_now, (z_b - z_p)))
        except Exception:
            v_left = v_right = None
        return d_left, d_right, v_left, v_right
    except Exception:
        return None, None, None, None


# Async queue mode helpers
def _queue_ensure_exec():
    return _bg_executor()


def _queue_add_one():
    # propose single z based on pair proposer (use first vector)
    try:
        vmc = st.session_state.get('vm_choice', 'DistanceHill')
        pp = 'CosineHill' if vmc == 'CosineHill' else 'DistanceHill'
        if pp == 'DistanceHill':
            Xd = yd = None
            try:
                with np.load(dataset_path_for_prompt(base_prompt)) as d:
                    Xd = d['X'] if 'X' in d.files else None
                    yd = d['y'] if 'y' in d.files else None
            except Exception:
                Xd = yd = None
            if (Xd is None or yd is None) and getattr(st.session_state, 'dataset_X', None) is not None:
                Xd = st.session_state.dataset_X
                yd = st.session_state.dataset_y
            from latent_logic import propose_pair_distancehill
            za, _ = propose_pair_distancehill(lstate, base_prompt, Xd, yd, alpha=float(alpha), gamma=0.5, trust_r=None)
        elif pp == 'CosineHill':
            Xd = yd = None
            try:
                with np.load(dataset_path_for_prompt(base_prompt)) as d:
                    Xd = d['X'] if 'X' in d.files else None
                    yd = d['y'] if 'y' in d.files else None
            except Exception:
                Xd = yd = None
            if (Xd is None or yd is None) and getattr(st.session_state, 'dataset_X', None) is not None:
                Xd = st.session_state.dataset_X
                yd = st.session_state.dataset_y
            from latent_logic import propose_pair_cosinehill
            za, _ = propose_pair_cosinehill(lstate, base_prompt, Xd, yd, alpha=float(alpha), beta=5.0, trust_r=None)
        else:
            za, _ = propose_next_pair(lstate, base_prompt, opts=_proposer_opts())
    except Exception:
        za = _sample_around_prompt(0.8)
    lat = z_to_latents(lstate, za)
    fut = bg.schedule_decode_latents(base_prompt, lat, lstate.width, lstate.height, steps, guidance_eff)
    item = {'z': za, 'future': fut, 'label': None}
    q = st.session_state.get('queue') or []
    q.append(item)
    st.session_state.queue = q


def _queue_fill_up_to():
    q = st.session_state.get('queue') or []
    st.session_state.queue = q
    # Fill up to desired size; allow multiple pending items to avoid stalls
    while len(st.session_state.queue) < int(queue_size):
        _queue_add_one()


def _queue_label(idx: int, label: int):
    q = st.session_state.get('queue') or []
    if 0 <= idx < len(q):
        z = q[idx]['z']
        _curation_add(int(label), z)
        # optionally refit from saved dataset for immediate model update
        try:
            _curation_train_and_next()  # trains from disk; does not alter queue
        except Exception:
            pass
        q.pop(idx)
        st.session_state.queue = q


def run_pair_mode() -> None:
    generate_pair()
    if st.button("Generate pair", type="primary"):
        set_model(selected_model)
        npf = st.session_state.get('next_prefetch')
        if npf and npf.get('f') and npf['f'].done():
            try:
                img_a, img_b = npf['f'].result()
                st.session_state.lz_pair = (npf['za'], npf['zb'])
                st.session_state.images = (img_a, img_b)
            except Exception:
                generate_pair()
            _prefetch_next_for_generate()
            if callable(st_rerun):
                st_rerun()
        else:
            generate_pair()
    img_left, img_right = st.session_state.images
    d_left, d_right, v_left, v_right = _pair_scores()
    _render_pair_ui(img_left, img_right, d_left, d_right, v_left, v_right)


def run_batch_mode() -> None:
    _curation_init_batch()
    _render_batch_ui()


def run_queue_mode() -> None:
    if 'queue' not in st.session_state:
        st.session_state.queue = []
    _queue_fill_up_to()
    _render_queue_ui()


# Run selected mode
# Without Pair mode: choose between Batch (default) and Async queue
try:
    async_queue_mode
except NameError:  # minimal guard for test stubs/import order
    async_queue_mode = False
if async_queue_mode:
    run_queue_mode()
else:
    run_batch_mode()

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
