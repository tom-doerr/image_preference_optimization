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
from persistence_ui import render_persistence_controls, render_metadata_panel
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
from latent_logic import ridge_fit  # for full-dataset training
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
        z1, z2 = propose_next_pair(new_state, st.session_state.prompt, opts=_proposer_opts())
        st.session_state.lz_pair = (z1, z2)
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

# Quick data strip at the very top of the sidebar
try:
    st.sidebar.subheader("Data")
    # Dataset rows from persisted NPZ
    try:
        _rows_cnt = int(dataset_rows_for_prompt(base_prompt))
    except Exception:
        _rows_cnt = 0
    # Simple train score using current ridge weights over any in-memory X/y
    try:
        X_ = getattr(lstate, 'X', None)
        y_ = getattr(lstate, 'y', None)
        if X_ is not None and y_ is not None and len(y_) > 0:
            _pred = X_ @ lstate.w
            _acc = float(np.mean(( _pred >= 0) == (y_ > 0)))
            _train_score = f"{_acc*100:.0f}%"
        else:
            _train_score = "n/a"
    except Exception:
        _train_score = "n/a"
    # Value model type and settings (best-effort; finalized below after sliders)
    try:
        cache = st.session_state.get('xgb_cache') or {}
        if cache.get('model') is not None:
            _vm_type = "XGBoost"
            _vm_settings = "n=50,depth=3"
        else:
            _vm_type = "Ridge"
            _vm_settings = f"λ={float(st.session_state.get('reg_lambda', 1e-2)):.3g}"
    except Exception:
        _vm_type, _vm_settings = "Ridge", "λ=1e-2"

    sidebar_metric_rows([("Dataset rows", _rows_cnt), ("Train score", _train_score)], per_row=2)
    sidebar_metric_rows([("Value model", _vm_type), ("Settings", _vm_settings)], per_row=2)
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

# Generation mode at top of sidebar (dropdown) — Pair mode removed
_gen_opts = ["Batch curation", "Async queue"]
_sb_sel = getattr(st.sidebar, 'selectbox', None)
selected_gen_mode = None
if callable(_sb_sel):
    try:
        # Default to Batch curation
        selected_gen_mode = _sb_sel("Generation mode", _gen_opts, index=1)
        if selected_gen_mode not in _gen_opts:
            selected_gen_mode = None
    except Exception:
        selected_gen_mode = None
# Simplified: hardcode sd-turbo; no model selector
_exp = getattr(st.sidebar, 'expander', None)
if callable(_exp):
    with _exp("Proposer controls", expanded=False):
        alpha = _sb_sld("Alpha (ridge d1)", 0.05, 3.0, value=0.5, step=0.05)
        beta = _sb_sld("Beta (ridge d2)", 0.05, 3.0, value=0.5, step=0.05)
        trust_r = _sb_sld("Trust radius (||y||)", 0.5, 5.0, value=2.5, step=0.1)
        lr_mu_ui = _sb_sld("Step size (lr_μ)", 0.05, 1.0, value=0.3, step=0.05)
        gamma_orth = _sb_sld("Orth explore (γ)", 0.0, 1.0, value=0.2, step=0.05)
        # Optional iterative controls (default disabled)
        iter_steps = _sb_sld("Optimization steps (latent)", 1, 10, value=1, step=1)
else:
    alpha = _sb_sld("Alpha (ridge d1)", 0.05, 3.0, value=0.5, step=0.05)
    beta = _sb_sld("Beta (ridge d2)", 0.05, 3.0, value=0.5, step=0.05)
    trust_r = _sb_sld("Trust radius (||y||)", 0.5, 5.0, value=2.5, step=0.1)
    lr_mu_ui = _sb_sld("Step size (lr_μ)", 0.05, 1.0, value=0.3, step=0.05)
    gamma_orth = _sb_sld("Orth explore (γ)", 0.0, 1.0, value=0.2, step=0.05)
    iter_steps = _sb_sld("Optimization steps (latent)", 1, 10, value=1, step=1)
# Value function option: Ridge (linear) vs XGBoost
use_xgb = st.sidebar.checkbox("Use XGBoost value function", value=False)

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
reg_lambda = _sb_sld("Ridge λ (regularization)", 1e-5, 1e-1, 1e-2)
try:
    st.session_state['reg_lambda'] = float(reg_lambda)
except Exception:
    pass
iter_eta = _sb_sld("Iterative step (eta)", 0.0, 1.0, 0.0, 0.05)
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
except Exception:
    pass

# State metadata panel
render_metadata_panel(st.session_state.state_path, st.session_state.prompt)

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
                value_scorer = lambda f: score_xgb_proba(mdl, f)
        except Exception:
            value_scorer = None
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
    # Always train from saved dataset on disk
    try:
        with np.load(dataset_path_for_prompt(base_prompt)) as d:
            X = d['X'] if 'X' in d.files else None
            y = d['y'] if 'y' in d.files else None
    except Exception:
        X = y = None
    # Prefer saved dataset; if missing, fall back to in-memory (keeps tests/light runs simple)
    if (X is None or y is None) and getattr(st.session_state, 'dataset_X', None) is not None:
        X = st.session_state.dataset_X
        y = st.session_state.dataset_y
    if X is not None and y is not None and getattr(X, 'shape', (0,))[0] > 0:
        try:
            lstate.w = ridge_fit(X, y, lam=float(reg_lambda))
        except Exception:
            pass
    _curation_new_batch()


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
    set_model(selected_model)
    z_p = z_from_prompt(lstate, base_prompt)
    feats_a = z_a - z_p
    feats_b = z_b - z_p
    winner = 'a' if side == 'a' else 'b'
    update_latent_ridge(lstate, z_a, z_b, winner, lr_mu=float(lr_mu_ui), lam=float(reg_lambda), feats_a=feats_a, feats_b=feats_b)
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
    for i, z_i in enumerate(st.session_state.cur_batch or []):
        lat = z_to_latents(lstate, z_i)
        img_i = generate_flux_image_latents(base_prompt, latents=lat, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff)
        _image_fragment(img_i, caption=f"Item {i}")
        if st.button(f"Good (+1) {i}", use_container_width=True):
            _curation_add(1, z_i)
            st.session_state.cur_labels[i] = 1
            _curation_replace_at(i)
            if callable(st_rerun):
                st_rerun()
        if st.button(f"Bad (-1) {i}", use_container_width=True):
            _curation_add(-1, z_i)
            st.session_state.cur_labels[i] = -1
            _curation_replace_at(i)
            if callable(st_rerun):
                st_rerun()
    if st.button("Train on dataset and next batch", type="primary"):
        _curation_train_and_next()
        if callable(st_rerun):
            st_rerun()


def _render_queue_ui() -> None:
    st.subheader("Async queue")
    q = st.session_state.get('queue') or []
    for i, it in enumerate(list(q)):
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
    # propose single z (use ridge first vector or random around prompt)
    try:
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
    # Generate only one image at a time: add a new one only if no pending futures
    pending = any((it.get('future') and not it['future'].done()) for it in q)
    if (not pending) and len(st.session_state.queue) < int(queue_size):
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
