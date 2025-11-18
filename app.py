import streamlit as st
import numpy as np
import os
import hashlib
from constants import (
    DEFAULT_PROMPT,
    MODEL_CHOICES,
    SMALL_VRAM_MAX_WIDTH,
    SMALL_VRAM_MAX_HEIGHT,
    SMALL_VRAM_MAX_STEPS,
)
from constants import Config
from env_info import get_env_summary
from ui import sidebar_metric_rows, render_pair_sidebar, env_panel, status_panel
from persistence import state_path_for_prompt, export_state_bytes
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
    # Always generate prompt via text path (simpler, reliable)
    try:
        st.session_state.prompt_image = generate_flux_image(
            st.session_state.prompt,
            width=new_state.width,
            height=new_state.height,
            steps=Config.DEFAULT_STEPS,
            guidance=Config.DEFAULT_GUIDANCE,
        )
    except Exception:
        st.session_state.prompt_image = None

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
flux_model = st.sidebar.selectbox("FLUX model", MODEL_CHOICES)
custom_model = st.sidebar.text_input("Custom HF model id (optional)", value="")
selected_model = custom_model.strip() or flux_model
if 'selected_model_override' in st.session_state:
    selected_model = st.session_state.selected_model_override
alpha = st.slider("Alpha (ridge d1)", 0.05, 3.0, 0.5, 0.05)
beta = st.slider("Beta (ridge d2)", 0.05, 3.0, 0.5, 0.05)
trust_r = st.slider("Trust radius (||y||)", 0.5, 5.0, 2.5, 0.1)
lr_mu_ui = st.slider("Step size (lr_μ)", 0.05, 1.0, 0.3, 0.05)
gamma_orth = st.slider("Orth explore (γ)", 0.0, 1.0, 0.2, 0.05)
# Optional iterative controls (default disabled)
iter_steps = st.slider("Iterative steps", 1, 10, 1, 1)
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

# Effective guidance: Turbo models prefer CFG=0.0
is_turbo = isinstance(selected_model, str) and ("sd-turbo" in selected_model or "sdxl-turbo" in selected_model)
guidance_eff = 0.0 if is_turbo else guidance

# Quick recipe: one-click Turbo defaults
if st.sidebar.button("Use Turbo defaults"):
    st.session_state.selected_model_override = "stabilityai/sd-turbo"
    _apply_state(init_latent_state(width=512, height=512))
    save_state(st.session_state.lstate, st.session_state.state_path)
    if callable(st_rerun):
        st_rerun()
if st.session_state.get('selected_model_override') == "stabilityai/sd-turbo":
    st.sidebar.caption("Turbo defaults active (sd-turbo • 512×512 • CFG 0.0)")

# (auto-run added after function definitions below)

mu_show = False

st.sidebar.subheader("Persistence")
_export_state_bytes = export_state_bytes  # back-compat for tests
render_persistence_controls(lstate, st.session_state.prompt, st.session_state.state_path, _apply_state, st_rerun)

# Autorun: always generate prompt image (text path), then generate the A/B pair if missing
set_model(selected_model)
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

env_panel(get_env_summary())

info = state_summary(lstate)
pairs_state = [("Latent dim", f"{info['d']}")]
pairs_state += [(k, f"{info[k]}") for k in ('width','height','step','sigma','mu_norm','w_norm','pairs_logged','choices_logged')]
sidebar_metric_rows(pairs_state, per_row=2)

# Debug panel: expose last pipeline call stats to spot black-frame issues
try:
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
    render_pair_sidebar(st.session_state.lstate, st.session_state.prompt, za, zb, lr_mu_val=float(lr_mu_ui))
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

def generate_pair():
    lat_a = z_to_latents(lstate, z_a)
    lat_b = z_to_latents(lstate, z_b)
    # Create best-effort placeholders if available (tests may stub them out)
    make_slot = getattr(st, 'empty', None)
    left_slot = make_slot() if callable(make_slot) else None
    right_slot = make_slot() if callable(make_slot) else None

    # Generate A
    # When injecting latents, use the state's width/height to avoid mismatches
    img_a = generate_flux_image_latents(base_prompt, latents=lat_a, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff)
    try:
        st.session_state.img_stats = st.session_state.get('img_stats') or {}
        st.session_state.img_stats['left'] = get_last_call().copy()
    except Exception:
        pass
    if hasattr(left_slot, 'image'):
        left_slot.image(img_a, caption="Left", use_container_width=True)
    st.session_state.images = (img_a, None)
    # Brief latent vector summary above the image (minimal; aids debugging)
    try:
        st.caption(f"z_left: first8={np.array2string(z_a[:8], precision=2, separator=', ')} | ‖z_l‖={float(np.linalg.norm(z_a)):.3f}")
    except Exception:
        pass

    # Generate B
    img_b = generate_flux_image_latents(base_prompt, latents=lat_b, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff)
    try:
        st.session_state.img_stats['right'] = get_last_call().copy()
    except Exception:
        pass
    if hasattr(right_slot, 'image'):
        right_slot.image(img_b, caption="Right", use_container_width=True)
    st.session_state.images = (img_a, img_b)
    try:
        st.caption(f"z_right: first8={np.array2string(z_b[:8], precision=2, separator=', ')} | ‖z_r‖={float(np.linalg.norm(z_b)):.3f}")
    except Exception:
        pass

    # μ preview removed


# history helpers removed

# Autorun: generate the pair once on load (keep it simple)
generate_pair()

if st.button("Generate pair", type="primary"):
    # Ensure model is loaded per selection
    set_model(selected_model)
    generate_pair()

img_left, img_right = st.session_state.images

left, right = st.columns(2)
# Compute prompt distances and predicted values for captions/metrics
try:
    z_p_cap = z_from_prompt(st.session_state.lstate, base_prompt)
    d_left = float(np.linalg.norm(z_a - z_p_cap))
    d_right = float(np.linalg.norm(z_b - z_p_cap))
    try:
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
    if img_left is not None:
        cap = f"Left (d_prompt={d_left:.3f})" if d_left is not None else "Left"
        st.image(img_left, caption=cap, use_container_width=True)
        # Show predicted value for Left (fallback to text if st.metric missing in tests)
        if v_left is not None:
            try:
                st.metric("V(left)", f"{v_left:.3f}")
            except Exception:
                st.write(f"V(left): {v_left:.3f}")
        # Content warning for left
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
        update_latent_ridge(lstate, z_a, z_b, 'a', lr_mu=float(lr_mu_ui), feats_a=feats_a, feats_b=feats_b)
        mode = 'iter' if (iter_steps > 1 or iter_eta > 0.0) else 'line'
        from latent_opt import ProposerOpts
        opts = ProposerOpts(mode=mode, trust_r=trust_r, gamma=gamma_orth, steps=int(iter_steps), eta=(float(iter_eta) if iter_eta > 0.0 else None))
        st.session_state.lz_pair = propose_next_pair(lstate, base_prompt, opts=opts)
        save_state(lstate, st.session_state.state_path)
        if callable(st_rerun):
            st_rerun()
with right:
    if img_right is not None:
        cap = f"Right (d_prompt={d_right:.3f})" if d_right is not None else "Right"
        st.image(img_right, caption=cap, use_container_width=True)
        if v_right is not None:
            try:
                st.metric("V(right)", f"{v_right:.3f}")
            except Exception:
                st.write(f"V(right): {v_right:.3f}")
        # Content warning for right
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
        update_latent_ridge(lstate, z_a, z_b, 'b', lr_mu=float(lr_mu_ui), feats_a=feats_a, feats_b=feats_b)
        mode = 'iter' if (iter_steps > 1 or iter_eta > 0.0) else 'line'
        from latent_opt import ProposerOpts
        opts = ProposerOpts(mode=mode, trust_r=trust_r, gamma=gamma_orth, steps=int(iter_steps), eta=(float(iter_eta) if iter_eta > 0.0 else None))
        st.session_state.lz_pair = propose_next_pair(lstate, base_prompt, opts=opts)
        save_state(lstate, st.session_state.state_path)
        if callable(st_rerun):
            st_rerun()

st.write(f"Interactions: {lstate.step}")
    # Best μ history UI removed
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
