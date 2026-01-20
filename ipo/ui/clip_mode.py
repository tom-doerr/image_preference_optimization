"""CLIP preference mode - train value model on image embeddings."""
import numpy as np
import streamlit as st
from ipo.infra.constants import Keys


def run_clip_mode():
    """Main entry point for CLIP preference mode."""
    from ipo.core.clip_db import init_db
    init_db()
    _init_state()
    st.header("CLIP Preference Mode")
    _render_sidebar()
    _render_gallery()


def _init_state():
    """Initialize session state for CLIP mode."""
    if Keys.CLIP_IMAGES not in st.session_state:
        st.session_state[Keys.CLIP_IMAGES] = []
    if Keys.CLIP_EMBEDS not in st.session_state:
        st.session_state[Keys.CLIP_EMBEDS] = []
    if Keys.CLIP_W not in st.session_state:
        st.session_state[Keys.CLIP_W] = None


def _render_sidebar():
    """Render sidebar controls."""
    st.sidebar.subheader("CLIP Settings")
    st.session_state[Keys.CLIP_MAX] = st.sidebar.slider("Max images", 4, 64, 16)
    if st.sidebar.button("Generate Batch"):
        _generate_batch()


def _gen_loop(prompt, n, mm, embed_fn):
    """Inner loop for generation."""
    for i in range(n):
        with st.spinner(f"Generating image {i+1}/{n}..."):
            img = mm.generate(prompt, seed=np.random.randint(1e9))
        with st.spinner(f"Embedding image {i+1}/{n}..."):
            emb = embed_fn(img)
        st.session_state[Keys.CLIP_IMAGES].append(img)
        st.session_state[Keys.CLIP_EMBEDS].append(emb)


def _generate_batch():
    """Generate images and embed them."""
    from ipo.core.model_manager import ModelManager
    from ipo.infra.clip_embed import embed_image
    prompt = st.session_state.get(Keys.PROMPT, "a photo")
    n = st.session_state.get(Keys.CLIP_MAX, 8)
    with st.spinner("Loading generation model..."):
        ModelManager.ensure_ready()
    _gen_loop(prompt, n, ModelManager, embed_image)


def _ridge_fit(X, y, lam=1.0):
    """Fit ridge regression, return weights."""
    if len(X) < 2:
        return None
    n, d = X.shape
    I = np.eye(d)
    return np.linalg.solve(X.T @ X + lam * I, X.T @ y)


def _img_to_bytes(img):
    """Convert PIL image to PNG bytes."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _save_rating(idx, label):
    """Save rating to DB and retrain model."""
    from ipo.core.clip_db import save_sample, get_samples
    emb = st.session_state[Keys.CLIP_EMBEDS][idx]
    img = st.session_state[Keys.CLIP_IMAGES][idx]
    prompt = st.session_state.get(Keys.PROMPT, "")
    save_sample(prompt, emb.astype(np.float32), label, _img_to_bytes(img))
    X, y = get_samples(prompt)
    if len(X) >= 2:
        st.session_state[Keys.CLIP_W] = _ridge_fit(X, y)


@st.fragment
def _rating_buttons(idx):
    """Rating buttons for a single image."""
    c1, c2 = st.columns(2)
    if c1.button("👍", key=f"up_{idx}"):
        _save_rating(idx, 1)
    if c2.button("👎", key=f"dn_{idx}"):
        _save_rating(idx, 0)


def _display_sorted(imgs):
    """Display images sorted by predicted score."""
    scores = _predict(st.session_state[Keys.CLIP_W])
    order = np.argsort(scores)[::-1]
    cols = st.columns(4)
    for i, idx in enumerate(order):
        with cols[i % 4]:
            st.image(imgs[idx], use_container_width=True)
            st.caption(f"Score: {scores[idx]:.2f}")
            _rating_buttons(idx)


def _predict(w):
    """Predict scores for current images."""
    if w is None or not st.session_state[Keys.CLIP_EMBEDS]:
        return [0.0] * len(st.session_state[Keys.CLIP_IMAGES])
    X = np.array(st.session_state[Keys.CLIP_EMBEDS])
    return (X @ w).tolist()


def _render_gallery():
    """Render image gallery with predicted scores."""
    imgs = st.session_state[Keys.CLIP_IMAGES]
    if not imgs:
        st.info("Click 'Generate Batch' to create images")
        return
    _display_sorted(imgs)
