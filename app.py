import streamlit as st
import numpy as np
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from constants import (
    DEFAULT_PROMPT,
    SMALL_VRAM_MAX_WIDTH,
    SMALL_VRAM_MAX_HEIGHT,
    SMALL_VRAM_MAX_STEPS,
)
from constants import Config
from env_info import get_env_summary
from ui import sidebar_metric_rows, render_pair_sidebar, env_panel, status_panel
from persistence import state_path_for_prompt, export_state_bytes, dataset_path_for_prompt
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
    generate_flux_image,  # text-only path
    set_model,
)
# Debug accessor (exists in flux_local)
from flux_local import get_last_call  # type: ignore

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

def _apply_state(new_state):
    """Apply a freshly loaded/created state to session and reset derived caches."""
    st.session_state.lstate = new_state
    # Initialize pair around the prompt anchor (symmetric)
    try:
        mode = 'iter' if (iter_steps > 1 or iter_eta > 0.0) else 'line'
        from latent_opt import ProposerOpts  # local import to avoid test stub issues
        opts = ProposerOpts(
            mode=mode,
            trust_r=trust_r,
            gamma=gamma_orth,
            steps=int(iter_steps),
            eta=(float(iter_eta) if iter_eta > 0.0 else None),
        )
        z1, z2 = propose_next_pair(new_state, st.session_state.prompt, opts=opts)
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

if 'lstate' not in st.session_state or prompt_changed:
    if os.path.exists(st.session_state.state_path):
        _apply_state(load_state(st.session_state.state_path))
    else:
        _apply_state(init_latent_state())
    if 'prompt_image' not in st.session_state:
        st.session_state.prompt_image = None

lstate = st.session_state.lstate
z_a, z_b = st.session_state.lz_pair
width = st.number_input("Width", min_value=256, max_value=1024, step=64, value=lstate.width)
height = st.number_input("Height", min_value=256, max_value=1024, step=64, value=lstate.height)
steps = st.slider("Steps", 1, 50, Config.DEFAULT_STEPS)
guidance = st.slider("Guidance", 0.0, 10.0, Config.DEFAULT_GUIDANCE, 0.1)
st.sidebar.header("Settings")
# Simplified: hardcode sd-turbo; no model selector
selected_model = "stabilityai/sd-turbo"
alpha = st.slider("Alpha (ridge d1)", 0.05, 3.0, 0.5, 0.05)
beta = st.slider("Beta (ridge d2)", 0.05, 3.0, 0.5, 0.05)
trust_r = st.slider("Trust radius (||y||)", 0.5, 5.0, 2.5, 0.1)
lr_mu_ui = st.slider("Step size (lr_μ)", 0.05, 1.0, 0.3, 0.05)
gamma_orth = st.slider("Orth explore (γ)", 0.0, 1.0, 0.2, 0.05)
# Optional iterative controls (default disabled)
iter_steps = st.slider("Optimization steps (latent)", 1, 10, 1, 1)
# Value function option: Ridge (linear) vs XGBoost
use_xgb = st.sidebar.checkbox("Use XGBoost value function", value=False)
curation_mode = st.sidebar.checkbox("Batch curation mode", value=False)
batch_size = st.sidebar.slider("Batch size", 2, 12, 6, 1)
reg_lambda = st.sidebar.slider("Ridge λ (regularization)", 1e-5, 1e-1, 1e-2)
iter_eta = st.slider("Iterative step (eta)", 0.0, 1.0, 0.0, 0.05)
use_clip = False

# 7 GB VRAM recipe: lighter model, smaller size, no CLIP
small_vram = st.sidebar.checkbox("7 GB VRAM mode", value=False)
if small_vram:
    selected_model = "runwayml/stable-diffusion-v1-5"
    width = min(int(width), SMALL_VRAM_MAX_WIDTH) if width is not None else SMALL_VRAM_MAX_WIDTH
    height = min(int(height), SMALL_VRAM_MAX_HEIGHT) if height is not None else SMALL_VRAM_MAX_HEIGHT
    steps = min(int(steps), SMALL_VRAM_MAX_STEPS) if steps is not None else SMALL_VRAM_MAX_STEPS
    pass

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
    st.session_state.prompt_image = generate_flux_image(
        base_prompt, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff
    )
    try:
        st.session_state.prompt_stats = get_last_call().copy()
    except Exception:
        st.session_state.prompt_stats = {}


# Prompt-only generation
st.subheader("Prompt-only generation")
if st.button("Generate from Prompt", use_container_width=True):
    set_model(selected_model)
    st.session_state.prompt_image = generate_flux_image(
        base_prompt, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff
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
    st.image(st.session_state.prompt_image, caption="Prompt-only image", use_container_width=True)
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
    sidebar_metric_rows(rows, per_row=2)
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

# Debug panel
if st.sidebar.checkbox("Debug", value=False):
    st.sidebar.subheader("Debug info")
    # Size alignment
    size_note = f"state_size={lstate.width}x{lstate.height} • slider_size={int(width)}x{int(height)} (latents decode uses state_size)"
    sidebar_metric_rows([("sizes", size_note)], per_row=1)
    # Last pipeline call info
    try:
        last = get_last_call()
        pairs = []
        if last.get('model_id'):
            pairs.append(("model_id", str(last['model_id'])))
        if last.get('event'):
            pairs.append(("event", str(last['event'])))
        if last.get('latents_std') is not None:
            pairs.append(("latents_std", f"{last['latents_std']:.3f}"))
        if last.get('latents_mean') is not None:
            pairs.append(("latents_mean", f"{last['latents_mean']:.3f}"))
        if last.get('width') and last.get('height'):
            pairs.append(("pipe_size", f"{last['width']}x{last['height']}"))
        if last.get('latents_shape'):
            pairs.append(("latents_shape", str(last['latents_shape'])))
        if pairs:
            sidebar_metric_rows(pairs, per_row=2)
    except Exception:
        pass
    # Heuristic warning if std looks degenerate
    try:
        ls = last.get('latents_std', None)
        if isinstance(ls, float) and ls < 1e-3:
            st.sidebar.write("warn: latents std≈0; try Reset or different prompt.")
    except Exception:
        pass
    # Sanity buttons
    if st.sidebar.button("Sanity: text-only decode"):
        img = generate_flux_image(base_prompt, width=256, height=256, steps=6, guidance=2.5)
        st.image(img, caption="Sanity text-only (256x256)", use_container_width=True)
    if st.sidebar.button("Sanity: random latents (256)"):
        try:
            h8, w8 = (256 // 8), (256 // 8)
            rnd = np.random.default_rng(0).standard_normal((1,4,h8,w8)).astype(np.float32)
            img = generate_flux_image_latents(base_prompt, latents=rnd, width=256, height=256, steps=6, guidance=2.5)
            st.image(img, caption="Sanity random latents (256x256)", use_container_width=True)
        except Exception as e:
            st.sidebar.write(f"random latents failed: {e}")

def _decode_one(side: str, latents, slot=None):
    """Decode one side and record last-call stats and optional streaming render."""
    img = generate_flux_image_latents(
        base_prompt,
        latents=latents,
        width=lstate.width,
        height=lstate.height,
        steps=steps,
        guidance=guidance_eff,
    )
    try:
        st.session_state.img_stats = st.session_state.get('img_stats') or {}
        st.session_state.img_stats[side] = get_last_call().copy()
    except Exception:
        pass
    if hasattr(slot, 'image'):
        slot.image(img, caption=side.capitalize(), use_container_width=True)
    return img


def generate_pair():
    lat_a = z_to_latents(lstate, z_a)
    lat_b = z_to_latents(lstate, z_b)
    # Best-effort placeholders if available (tests may stub them out)
    make_slot = getattr(st, 'empty', None)
    left_slot = make_slot() if callable(make_slot) else None
    right_slot = make_slot() if callable(make_slot) else None

    # Decode sequentially with shared helper
    img_a = _decode_one('left', lat_a, left_slot)
    st.session_state.images = (img_a, None)
    try:
        st.caption(
            f"z_left: first8={np.array2string(z_a[:8], precision=2, separator=', ')} | ‖z_l‖={float(np.linalg.norm(z_a)):.3f}"
        )
    except Exception:
        pass

    img_b = _decode_one('right', lat_b, right_slot)
    st.session_state.images = (img_a, img_b)
    try:
        st.caption(
            f"z_right: first8={np.array2string(z_b[:8], precision=2, separator=', ')} | ‖z_r‖={float(np.linalg.norm(z_b)):.3f}"
        )
    except Exception:
        pass
    # Prefetch next pair for the Generate button
    _prefetch_next_for_generate()


def _ensure_executor():
    ex = st.session_state.get('_prefetch_exec')
    if ex is None:
        ex = ThreadPoolExecutor(max_workers=2)
        st.session_state._prefetch_exec = ex
    return ex


def _prefetch_next_for_generate():
    try:
        try:
            mode = 'iter' if (iter_steps > 1 or iter_eta > 0.0) else 'line'
            from latent_opt import ProposerOpts
            opts = ProposerOpts(mode=mode, trust_r=trust_r, gamma=gamma_orth, steps=int(iter_steps), eta=(float(iter_eta) if iter_eta > 0.0 else None))
            za_n, zb_n = propose_next_pair(lstate, base_prompt, opts=opts)
        except Exception:
            za_n, zb_n = propose_latent_pair_ridge(lstate)
        la_n = z_to_latents(lstate, za_n)
        lb_n = z_to_latents(lstate, zb_n)
        ex = _ensure_executor()
        fa = ex.submit(
            generate_flux_image_latents,
            base_prompt,
            lat_a := la_n,
            lstate.width,
            lstate.height,
            steps,
            guidance_eff,
        )
        fb = ex.submit(
            generate_flux_image_latents,
            base_prompt,
            lat_b := lb_n,
            lstate.width,
            lstate.height,
            steps,
            guidance_eff,
        )
        st.session_state.next_prefetch = {
            'za': za_n,
            'zb': zb_n,
            'fa': fa,
            'fb': fb,
        }
    except Exception:
        st.session_state.pop('next_prefetch', None)

    # μ preview removed


# history helpers removed

def _curation_init_batch():
    if st.session_state.get('cur_batch') is None:
        st.session_state.cur_batch = []
        st.session_state.cur_labels = []
        _curation_new_batch()


def _curation_new_batch():
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


def _curation_add(label: int, z: np.ndarray):
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
        path = dataset_path_for_prompt(base_prompt)
        try:
            with np.load(path) as d:
                Xd = d['X'] if 'X' in d.files else np.zeros((0, feat.shape[1]))
                yd = d['y'] if 'y' in d.files else np.zeros((0,))
        except FileNotFoundError:
            Xd, yd = np.zeros((0, feat.shape[1])), np.zeros((0,))
        X_new = np.vstack([Xd, feat]) if Xd.size else feat
        y_new = np.concatenate([yd, lab]) if yd.size else lab
        np.savez_compressed(path, X=X_new, y=y_new)
    except Exception:
        pass


def _curation_train_and_next():
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


# Autorun: generate the pair once on load (keep it simple)
if not curation_mode:
    generate_pair()
else:
    _curation_init_batch()

if st.button("Generate pair", type="primary"):
    # Ensure model is loaded per selection
    set_model(selected_model)
    npf = st.session_state.get('next_prefetch')
    if npf and npf.get('fa') and npf.get('fb') and npf['fa'].done() and npf['fb'].done():
        try:
            img_a = npf['fa'].result()
            img_b = npf['fb'].result()
            st.session_state.lz_pair = (npf['za'], npf['zb'])
            st.session_state.images = (img_a, img_b)
        except Exception:
            generate_pair()
        # Chain another prefetch
        _prefetch_next_for_generate()
        if callable(st_rerun):
            st_rerun()
    else:
        generate_pair()

img_left, img_right = st.session_state.images

left, right = st.columns(2)
# Compute prompt distances and predicted values for captions/metrics
try:
    z_p_cap = z_from_prompt(st.session_state.lstate, base_prompt)
    d_left = float(np.linalg.norm(z_a - z_p_cap))
    d_right = float(np.linalg.norm(z_b - z_p_cap))
    try:
        if use_xgb:
            cache = st.session_state.get('xgb_cache') or {}
            mdl = cache.get('model')
            if mdl is not None:
                from xgb_value import score_xgb_proba  # type: ignore
                v_left = score_xgb_proba(mdl, (z_a - z_p_cap))
                v_right = score_xgb_proba(mdl, (z_b - z_p_cap))
            else:
                v_left = v_right = None
        else:
            w_now = st.session_state.lstate.w
            v_left = float(np.dot(w_now, (z_a - z_p_cap)))
            v_right = float(np.dot(w_now, (z_b - z_p_cap)))
    except Exception:
        v_left = v_right = None
except Exception:
    z_p_cap = None
    d_left = d_right = None
    v_left = v_right = None
with left:
    if not curation_mode:
        if img_left is not None:
            cap = f"Left (d_prompt={d_left:.3f})" if d_left is not None else "Left"
            st.image(img_left, caption=cap, use_container_width=True)
            if v_left is not None:
                try:
                    st.metric("V(left)", f"{v_left:.3f}")
                except Exception:
                    st.write(f"V(left): {v_left:.3f}")
            try:
                stats = (st.session_state.get('img_stats') or {}).get('left') or {}
                s = stats.get('img0_std')
                if s is not None and float(s) < 2.0:
                    (getattr(st, 'warning', None) or st.write)(f"⚠️ Low content (std={float(s):.2f})")
            except Exception:
                pass
        if st.button("Prefer Left", use_container_width=True):
            set_model(selected_model)
            z_p = z_from_prompt(lstate, base_prompt)
            feats_a = z_a - z_p
            feats_b = z_b - z_p
            # Update state (mu, history), then always refit from saved dataset
            update_latent_ridge(lstate, z_a, z_b, 'a', lr_mu=float(lr_mu_ui), lam=float(reg_lambda), feats_a=feats_a, feats_b=feats_b)
            # Append both items to dataset and retrain from disk
            try:
                _curation_add(1, z_a)
                _curation_add(-1, z_b)
                _curation_train_and_next()  # train (does not change current pair)
            except Exception:
                pass
            mode = 'iter' if (iter_steps > 1 or iter_eta > 0.0) else 'line'
            from latent_opt import ProposerOpts
            opts = ProposerOpts(mode=mode, trust_r=trust_r, gamma=gamma_orth, steps=int(iter_steps), eta=(float(iter_eta) if iter_eta > 0.0 else None))
            st.session_state.lz_pair = propose_next_pair(lstate, base_prompt, opts=opts)
            save_state(lstate, st.session_state.state_path)
            if callable(st_rerun):
                st_rerun()
    else:
        st.subheader("Curation batch")
        # Render batch items vertically (simple and clear)
        for i, z_i in enumerate(st.session_state.cur_batch or []):
            lat = z_to_latents(lstate, z_i)
            img_i = generate_flux_image_latents(base_prompt, latents=lat, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff)
            st.image(img_i, caption=f"Item {i}", use_container_width=True)
            cols = st.columns(2)
            with cols[0]:
                if st.button(f"Accept {i}"):
                    _curation_add(1, z_i)
                    st.session_state.cur_labels[i] = 1
            with cols[1]:
                if st.button(f"Reject {i}"):
                    _curation_add(-1, z_i)
                    st.session_state.cur_labels[i] = -1
        if st.button("Train on dataset and next batch", type="primary"):
            _curation_train_and_next()
            if callable(st_rerun):
                st_rerun()
with right:
    if not curation_mode:
        if img_right is not None:
            cap = f"Right (d_prompt={d_right:.3f})" if d_right is not None else "Right"
            st.image(img_right, caption=cap, use_container_width=True)
            if v_right is not None:
                try:
                    st.metric("V(right)", f"{v_right:.3f}")
                except Exception:
                    st.write(f"V(right): {v_right:.3f}")
            try:
                stats = (st.session_state.get('img_stats') or {}).get('right') or {}
                s = stats.get('img0_std')
                if s is not None and float(s) < 2.0:
                    (getattr(st, 'warning', None) or st.write)(f"⚠️ Low content (std={float(s):.2f})")
            except Exception:
                pass
        if st.button("Prefer Right", use_container_width=True):
            set_model(selected_model)
            z_p = z_from_prompt(lstate, base_prompt)
            feats_a = z_a - z_p
            feats_b = z_b - z_p
            update_latent_ridge(lstate, z_a, z_b, 'b', lr_mu=float(lr_mu_ui), lam=float(reg_lambda), feats_a=feats_a, feats_b=feats_b)
            try:
                _curation_add(-1, z_a)
                _curation_add(1, z_b)
                _curation_train_and_next()
            except Exception:
                pass
            mode = 'iter' if (iter_steps > 1 or iter_eta > 0.0) else 'line'
            from latent_opt import ProposerOpts
            opts = ProposerOpts(mode=mode, trust_r=trust_r, gamma=gamma_orth, steps=int(iter_steps), eta=(float(iter_eta) if iter_eta > 0.0 else None))
            st.session_state.lz_pair = propose_next_pair(lstate, base_prompt, opts=opts)
            save_state(lstate, st.session_state.state_path)
            if callable(st_rerun):
                st_rerun()

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
