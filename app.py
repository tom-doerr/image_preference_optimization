import streamlit as st
import numpy as np
import os
import hashlib
from datetime import datetime, timezone
import io
from constants import (
    APP_VERSION,
    DEFAULT_PROMPT,
    MODEL_CHOICES,
    SMALL_VRAM_MAX_WIDTH,
    SMALL_VRAM_MAX_HEIGHT,
    SMALL_VRAM_MAX_STEPS,
)
from env_info import get_env_summary
from metrics import pair_metrics
from latent_opt import (
    init_latent_state,
    propose_latent_pair_ridge,
    propose_next_pair,
    z_to_latents,
    z_from_prompt,
    update_latent_ridge,
    save_state,
    load_state,
    dumps_state,
    loads_state,
    state_summary,
)
from flux_local import (
    generate_flux_image_latents as generate_flux_image,
    set_model,
)

# Optional text-only generator (tests often stub only the latents path)
try:  # minimal compat for tests; not a runtime fallback
    from flux_local import generate_flux_image as _text_image_fn  # type: ignore
except Exception:  # pragma: no cover - missing in some test stubs
    _text_image_fn = None

st.set_page_config(page_title="Latent Preference Optimizer", layout="centered")
# Streamlit rerun API shim: prefer st.rerun(), fallback to experimental in older versions
st_rerun = getattr(st, 'rerun', getattr(st, 'experimental_rerun', None))
st.title("Optimize Latents by Preference — FLUX (Local GPU)")
st.caption("Fixed prompt; we optimize the latent vector directly from your choices.")

def _state_path_for_prompt(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:10]
    return f"latent_state_{h}.npz"

# Prompt-aware persistence
if 'prompt' not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT

base_prompt = st.text_input("Prompt", value=st.session_state.prompt)
prompt_changed = (base_prompt != st.session_state.prompt)
if prompt_changed:
    st.session_state.prompt = base_prompt

st.session_state.state_path = _state_path_for_prompt(st.session_state.prompt)

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
steps = st.slider("Steps", 4, 50, 8)
guidance = st.slider("Guidance", 1.0, 10.0, 3.5, 0.1)
st.sidebar.header("Settings")
flux_model = st.sidebar.selectbox("FLUX model", MODEL_CHOICES)
custom_model = st.sidebar.text_input("Custom HF model id (optional)", value="")
selected_model = custom_model.strip() or flux_model
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

# (auto-run added after function definitions below)

st.sidebar.subheader("μ Preview")
mu_show = st.sidebar.checkbox("Show μ preview", value=True)

st.sidebar.subheader("Persistence")

def _export_state_bytes(state, prompt: str) -> bytes:
    raw = dumps_state(state)
    with np.load(io.BytesIO(raw)) as data:
        items = {k: data[k] for k in data.files}
    items['prompt'] = np.array(prompt)
    items['created_at'] = np.array(datetime.now(timezone.utc).isoformat())
    items['app_version'] = np.array(APP_VERSION)
    buf = io.BytesIO()
    np.savez_compressed(buf, **items)
    return buf.getvalue()

st.sidebar.download_button(
    label="Download state (.npz)",
    data=_export_state_bytes(lstate, st.session_state.prompt),
    file_name="latent_state.npz",
    mime="application/octet-stream",
)
uploaded = st.sidebar.file_uploader("Upload state (.npz)", type=["npz"])
if uploaded is not None and st.sidebar.button("Load uploaded state"):
    data_bytes = uploaded.read()
    up_prompt = None
    try:
        arr = np.load(io.BytesIO(data_bytes))
        if 'prompt' in arr.files:
            up_prompt = arr['prompt'].item()
    except Exception:
        up_prompt = None
    if up_prompt is not None and up_prompt != st.session_state.prompt:
        st.sidebar.warning(f"Uploaded state is for a different prompt: '{up_prompt}'. Change the Prompt or switch via Manage states, then load.")
        if st.sidebar.button("Switch to uploaded prompt and load now"):
            # Switch prompt and load uploaded state immediately
            st.session_state.prompt = up_prompt
            st.session_state.state_path = _state_path_for_prompt(up_prompt)
            new_state = loads_state(data_bytes)
            st.session_state.lstate = new_state
            st.session_state.lz_pair = propose_latent_pair_ridge(new_state)
            st.session_state.images = (None, None)
            st.session_state.mu_image = None
            if getattr(new_state, 'mu_hist', None) is not None and new_state.mu_hist.size > 0:
                st.session_state.mu_history = [m.copy() for m in new_state.mu_hist]
            else:
                st.session_state.mu_history = [new_state.mu.copy()]
            st.session_state.mu_best_idx = 0
            save_state(new_state, st.session_state.state_path)
            if callable(st_rerun):
                st_rerun()
    else:
        new_state = loads_state(data_bytes)
        _apply_state(new_state)
        save_state(new_state, st.session_state.state_path)
        if callable(st_rerun):
            st_rerun()

# Autorun: generate prompt image first (once), then generate the A/B pair (once)
set_model(selected_model)
if st.session_state.get('prompt_image') is None:
    if _text_image_fn is not None:
        st.session_state.prompt_image = _text_image_fn(
            base_prompt, width=width, height=height, steps=steps, guidance=guidance
        )
    else:
        z0 = z_from_prompt(st.session_state.lstate, base_prompt)
        lat0 = z_to_latents(st.session_state.lstate, z0)
        st.session_state.prompt_image = generate_flux_image(
            base_prompt, latents=lat0, width=width, height=height, steps=steps, guidance=guidance
        )


# Prompt-only generation
st.subheader("Prompt-only generation")
if st.button("Generate from Prompt", use_container_width=True):
    set_model(selected_model)
    if _text_image_fn is not None:
        st.session_state.prompt_image = _text_image_fn(
            base_prompt, width=width, height=height, steps=steps, guidance=guidance
        )
    else:
        # Minimal deterministic prompt seed → latents decode
        z = z_from_prompt(st.session_state.lstate, base_prompt)
        lat = z_to_latents(st.session_state.lstate, z)
        st.session_state.prompt_image = generate_flux_image(
            base_prompt, latents=lat, width=width, height=height, steps=steps, guidance=guidance
        )
if st.session_state.get('prompt_image') is not None:
    st.image(st.session_state.prompt_image, caption="Prompt-only image", use_container_width=True)

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

# Consolidated sidebar metric helpers
def _sb_metric(label: str, value) -> None:
    try:
        if callable(getattr(st.sidebar, 'metric', None)):
            st.sidebar.metric(label, str(value))
        else:
            st.sidebar.write(f"{label}: {value}")
    except Exception:
        st.sidebar.write(f"{label}: {value}")


def _sb_metric_rows(pairs, per_row: int = 2) -> None:
    try:
        for i in range(0, len(pairs), per_row):
            row = pairs[i:i + per_row]
            if hasattr(st.sidebar, 'columns') and callable(getattr(st.sidebar, 'columns', None)) and len(row) > 1:
                cols = st.sidebar.columns(len(row))
                for (label, value), col in zip(row, cols):
                    with col:
                        _sb_metric(label, value)
            else:
                for label, value in row:
                    _sb_metric(label, value)
    except Exception:
        for label, value in pairs:
            _sb_metric(label, value)


def _render_pair_sidebar(lstate, prompt: str, za: np.ndarray, zb: np.ndarray, lr_mu_val: float) -> None:
    """Render vector info, prompt distances, predicted values, and step sizes for current pair."""
    w = lstate.w
    m = pair_metrics(w, za, zb)
    st.sidebar.subheader("Vector info (current pair)")
    _sb_metric_rows([
        ("‖z_a‖", f"{m['za_norm']:.3f}"),
        ("‖z_b‖", f"{m['zb_norm']:.3f}"),
        ("‖z_b−z_a‖", f"{m['diff_norm']:.3f}")
    ], per_row=2)
    cos = m['cos_w_diff']
    _sb_metric_rows([("cos(w, z_b−z_a)", "n/a" if (cos is None or not np.isfinite(float(cos))) else f"{float(cos):.3f}")], per_row=1)
    z_p = z_from_prompt(lstate, prompt)
    _sb_metric_rows([
        ("‖μ−z_prompt‖", f"{float(np.linalg.norm(lstate.mu - z_p)):.3f}"),
        ("‖z_a−z_prompt‖", f"{float(np.linalg.norm(za - z_p)):.3f}"),
        ("‖z_b−z_prompt‖", f"{float(np.linalg.norm(zb - z_p)):.3f}")
    ], per_row=2)
    v_left = float(np.dot(w, (za - z_p)))
    v_right = float(np.dot(w, (zb - z_p)))
    _sb_metric_rows([("V(left)", f"{v_left:.3f}"), ("V(right)", f"{v_right:.3f}")], per_row=2)
    mu = lstate.mu
    _sb_metric_rows([
        ("step(A)", f"{lr_mu_val * float(np.linalg.norm(za - mu)):.3f}"),
        ("step(B)", f"{lr_mu_val * float(np.linalg.norm(zb - mu)):.3f}")
    ], per_row=2)
# Environment/version
st.sidebar.subheader("Environment")
env = get_env_summary()
_metric = getattr(st.sidebar, 'metric', None)
if callable(_metric):
    _metric("Python", f"{env['python']}")
    _metric("torch/CUDA", f"{env['torch']} | {env['cuda']}")
    if env.get('streamlit') and env['streamlit'] not in ('unknown', 'not imported'):
        _metric("Streamlit", f"{env['streamlit']}")
else:
    st.sidebar.write(f"Python: {env['python']}")
    st.sidebar.write(f"torch: {env['torch']} | CUDA: {env['cuda']}")
    if env.get('streamlit') and env['streamlit'] not in ('unknown', 'not imported'):
        st.sidebar.write(f"Streamlit: {env['streamlit']}")
info = state_summary(lstate)
st.sidebar.write(f"Latent dim: {info['d']}")
for k in ('width','height','step','sigma','mu_norm','w_norm','pairs_logged','choices_logged'):
    st.sidebar.write(f"{k}: {info[k]}")
if callable(_metric):
    _metric("Latent dim", f"{info['d']}")
    for k in ('width','height','step','sigma','mu_norm','w_norm','pairs_logged','choices_logged'):
        _metric(k, f"{info[k]}")

# State metadata (if present on disk)
try:
    if os.path.exists(st.session_state.state_path):
        with np.load(st.session_state.state_path) as data:
            meta_version = data['app_version'].item() if 'app_version' in data.files else None
            meta_created = data['created_at'].item() if 'created_at' in data.files else None
        if meta_version or meta_created:
            st.sidebar.subheader("State metadata")
            if meta_version:
                st.sidebar.write(f"app_version: {meta_version}")
            if meta_created:
                st.sidebar.write(f"created_at: {meta_created}")
            if callable(_metric):
                if meta_version:
                    _metric("app_version", f"{meta_version}")
                if meta_created:
                    _metric("created_at", f"{meta_created}")
except Exception:
    pass

# Images status (concise)
st.sidebar.subheader("Images status")
left_ready = 'ready' if st.session_state.images[0] is not None else 'empty'
right_ready = 'ready' if st.session_state.images[1] is not None else 'empty'
mu_ready = 'ready' if st.session_state.mu_image is not None else 'empty'
st.sidebar.write(f"Left: {left_ready}")
st.sidebar.write(f"Right: {right_ready}")
st.sidebar.write(f"μ preview: {mu_ready}")
if callable(_metric):
    _metric("Left", left_ready)
    _metric("Right", right_ready)
    _metric("μ preview", mu_ready)

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
        st.session_state.state_path = _state_path_for_prompt(sel)
        if os.path.exists(st.session_state.state_path):
            _apply_state(load_state(st.session_state.state_path))
        else:
            _apply_state(init_latent_state())
        if callable(st_rerun):
            st_rerun()

def generate_pair():
    lat_a = z_to_latents(lstate, z_a)
    lat_b = z_to_latents(lstate, z_b)
    lat_m = z_to_latents(lstate, lstate.mu)
    # Create best-effort placeholders if available (tests may stub them out)
    make_slot = getattr(st, 'empty', None)
    left_slot = make_slot() if callable(make_slot) else None
    right_slot = make_slot() if callable(make_slot) else None
    mu_slot = make_slot() if callable(make_slot) else None

    # Generate A
    img_a = generate_flux_image(base_prompt, latents=lat_a, width=width, height=height, steps=steps, guidance=guidance)
    if hasattr(left_slot, 'image'):
        left_slot.image(img_a, caption="Left", use_container_width=True)
    st.session_state.images = (img_a, None)

    # Generate B
    img_b = generate_flux_image(base_prompt, latents=lat_b, width=width, height=height, steps=steps, guidance=guidance)
    if hasattr(right_slot, 'image'):
        right_slot.image(img_b, caption="Right", use_container_width=True)
    st.session_state.images = (img_a, img_b)

    # Generate μ preview; always compute, only display when enabled
    img_m = generate_flux_image(base_prompt, latents=lat_m, width=width, height=height, steps=steps, guidance=guidance)
    if hasattr(mu_slot, 'image') and mu_show:
        mu_slot.image(img_m, caption="Current μ decode", use_container_width=True)
    st.session_state.mu_image = img_m


def _update_history():
    lstate = st.session_state.lstate
    st.session_state.mu_history.append(lstate.mu.copy())
    # mirror into persisted state
    if getattr(lstate, 'mu_hist', None) is None:
        lstate.mu_hist = np.array([st.session_state.mu_history[-1]], dtype=float)
    else:
        lstate.mu_hist = np.vstack([lstate.mu_hist, st.session_state.mu_history[-1]])
    w = lstate.w
    # Choose best by current w·mu across snapshots
    scores = [float(np.dot(w, mu)) for mu in st.session_state.mu_history]
    st.session_state.mu_best_idx = int(np.argmax(scores))

# Autorun: if pair images not yet generated, do it now (after function is defined)
if st.session_state.images == (None, None):
    generate_pair()

if st.button("Generate pair", type="primary"):
    # Ensure model is loaded per selection
    set_model(selected_model)
    generate_pair()

img_left, img_right = st.session_state.images

left, right = st.columns(2)
# Compute prompt distances for captions
try:
    z_p_cap = z_from_prompt(st.session_state.lstate, base_prompt)
    d_left = float(np.linalg.norm(z_a - z_p_cap))
    d_right = float(np.linalg.norm(z_b - z_p_cap))
except Exception:
    z_p_cap = None
    d_left = d_right = None
with left:
    if img_left is not None:
        cap = f"Left (d_prompt={d_left:.3f})" if d_left is not None else "Left"
        st.image(img_left, caption=cap, use_container_width=True)
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
        _update_history()
        save_state(lstate, st.session_state.state_path)
        if callable(st_rerun):
            st_rerun()
with right:
    if img_right is not None:
        cap = f"Right (d_prompt={d_right:.3f})" if d_right is not None else "Right"
        st.image(img_right, caption=cap, use_container_width=True)
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
        _update_history()
        save_state(lstate, st.session_state.state_path)
        if callable(st_rerun):
            st_rerun()

st.write(f"Interactions: {lstate.step}")
st.subheader("Best μ (history)")
st.write(f"Snapshots: {len(st.session_state.mu_history)}  |  Best index: {st.session_state.mu_best_idx}")
if st.button("Revert to Best μ"):
    idx = st.session_state.mu_best_idx
    lstate.mu = st.session_state.mu_history[idx].copy()
    # Refresh μ preview
    lat_m = z_to_latents(lstate, lstate.mu)
    st.session_state.mu_image = generate_flux_image(base_prompt, latents=lat_m, width=width, height=height, steps=steps, guidance=guidance)
    # Propose next pair
    st.session_state.lz_pair = propose_latent_pair_ridge(lstate, alpha=alpha, beta=beta, trust_r=trust_r)
    save_state(lstate, st.session_state.state_path)
    if callable(st_rerun):
        st_rerun()
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

# Vector info for current pair
try:
    za, zb = st.session_state.lz_pair
    _render_pair_sidebar(st.session_state.lstate, base_prompt, za, zb, lr_mu_val=float(lr_mu_ui))
except Exception:
    pass
if mu_show:
    st.subheader("Greedy preview (μ)")
    if st.session_state.mu_image is not None:
        st.image(st.session_state.mu_image, caption="Current μ decode", use_container_width=True)
    else:
        if st.button("Preview μ now"):
            lat_m = z_to_latents(st.session_state.lstate, st.session_state.lstate.mu)
            st.session_state.mu_image = generate_flux_image(base_prompt, latents=lat_m, width=st.session_state.lstate.width, height=st.session_state.lstate.height, steps=20, guidance=3.5)
            if callable(st_rerun):
                st_rerun()
