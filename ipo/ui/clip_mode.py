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

    left_col, right_col = st.columns([1, 3])
    with left_col:
        st.subheader("Current")
        gen_box = st.empty()
        _render_stats()
    with right_col:
        st.subheader("Rate")
        _render_sorted_gallery()

    with left_col:
        if st.button("Generate", key="clip_gen_btn"):
            _generate_batch(gen_box)


def _fit_and_store(X, y):
    """Fit ridge with CV and store weights + stats."""
    if len(X) < 2:
        st.session_state[Keys.CLIP_W] = None
        st.session_state[Keys.CLIP_ALPHA] = None
        st.session_state[Keys.CLIP_CV_SCORES] = {}
        return
    w, alpha, scores = _ridge_cv(X, y)
    st.session_state[Keys.CLIP_W] = w
    st.session_state[Keys.CLIP_ALPHA] = alpha
    st.session_state[Keys.CLIP_CV_SCORES] = scores


def _load_weights_from_db():
    """Load trained weights from DB if samples exist."""
    from ipo.core.clip_db import get_samples
    prompt = st.session_state.get(Keys.PROMPT, "")
    X, y = get_samples(prompt)
    _fit_and_store(X, y)


def _init_state():
    """Initialize session state for CLIP mode."""
    if Keys.CLIP_IMAGES not in st.session_state:
        st.session_state[Keys.CLIP_IMAGES] = []
    if Keys.CLIP_EMBEDS not in st.session_state:
        st.session_state[Keys.CLIP_EMBEDS] = []
    if Keys.CLIP_W not in st.session_state:
        _load_weights_from_db()
    if "clip_generating" not in st.session_state:
        st.session_state["clip_generating"] = False


def _render_sidebar():
    """Render sidebar controls."""
    st.sidebar.subheader("CLIP Settings")
    st.session_state[Keys.CLIP_MAX] = st.sidebar.slider("Max images", 4, 64, 16)


def _render_stats():
    """Show sample counts and model stats."""
    from ipo.core.clip_db import get_samples
    prompt = st.session_state.get(Keys.PROMPT, "")
    X, y = get_samples(prompt)
    n_pos = int((y > 0).sum()) if len(y) else 0
    n_neg = len(y) - n_pos
    st.markdown(f"**Samples:** {len(X)} (+{n_pos}/-{n_neg})")
    alpha = st.session_state.get(Keys.CLIP_ALPHA)
    cv = st.session_state.get(Keys.CLIP_CV_SCORES) or {}
    if alpha:
        st.markdown(f"**Best α:** {alpha}")
    if cv:
        for a, r2 in sorted(cv.items()):
            m = "**" if a == alpha else ""
            st.text(f"{m}{a}{m}: R²={r2:.3f}")


def _score_one(emb):
    """Score a single embedding with current weights."""
    w = st.session_state.get(Keys.CLIP_W)
    return float(np.dot(emb, w)) if w is not None else 0.0


def _gen_loop(prompt, n, mm, embed_fn, placeholder):
    """Generate images, show current in placeholder."""
    for i in range(n):
        with placeholder.container():
            st.text(f"Generating {i+1}/{n}...")
        img = mm.generate(prompt, seed=np.random.randint(1e9))
        emb = embed_fn(img)
        score = _score_one(emb)
        st.session_state[Keys.CLIP_IMAGES].append(img)
        st.session_state[Keys.CLIP_EMBEDS].append(emb)
        with placeholder.container():
            st.image(img, caption=f"s={score:.2f}", use_container_width=True)
    st.session_state["clip_generating"] = False
    st.rerun()


def _generate_batch(placeholder):
    """Generate images and embed them."""
    from ipo.core.model_manager import ModelManager
    from ipo.infra.clip_embed import embed_image, load_siglip
    prompt = st.session_state.get(Keys.PROMPT, "a photo")
    n = st.session_state.get(Keys.CLIP_MAX, 8)
    with st.spinner("Loading generation model..."):
        ModelManager.ensure_ready()
    with st.spinner("Loading SigLIP..."):
        load_siglip()
    _gen_loop(prompt, n, ModelManager, embed_image, placeholder)


ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def _ridge_fit(X, y, alpha=1.0):
    """Fit ridge with specific alpha."""
    if len(X) < 2:
        return None
    I = np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + alpha * I, X.T @ y)


def _ridge_cv(X, y):
    """CV for best alpha. Returns (w, alpha, {alpha: r2})."""
    if len(X) < 4:
        return _ridge_fit(X, y), 1.0, {}
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    cv = max(2, min(3, len(X) // 2))
    scores = {}
    for a in ALPHAS:
        try:
            s = np.nanmean(cross_val_score(Ridge(a), X, y, cv=cv))
            scores[a] = s if np.isfinite(s) else 0.0
        except Exception:
            scores[a] = 0.0
    best = max(scores, key=scores.get)
    return _ridge_fit(X, y, best), best, scores


def _img_to_bytes(img):
    """Convert PIL image to PNG bytes."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _vote_and_remove(idx, label):
    """Save rating, remove from lists, retrain."""
    from ipo.core.clip_db import save_sample, get_samples
    emb = st.session_state[Keys.CLIP_EMBEDS][idx]
    img = st.session_state[Keys.CLIP_IMAGES][idx]
    prompt = st.session_state.get(Keys.PROMPT, "")
    save_sample(prompt, emb.astype(np.float32), label, _img_to_bytes(img))
    st.session_state[Keys.CLIP_IMAGES].pop(idx)
    st.session_state[Keys.CLIP_EMBEDS].pop(idx)
    X, y = get_samples(prompt)
    _fit_and_store(X, y)


@st.fragment
def _render_sorted_gallery():
    """Fragment: sorted gallery with vote buttons."""
    imgs = st.session_state[Keys.CLIP_IMAGES]
    if not imgs:
        return
    scores = _predict(st.session_state[Keys.CLIP_W])
    order = np.argsort(scores)[::-1]
    for idx in order:
        c1, c2 = st.columns([4, 1])
        c1.image(imgs[idx], width=150)
        c1.write(f"s={scores[idx]:.2f}")
        if c2.button("👍", key=f"u{idx}"):
            _vote_and_remove(idx, 1)
            st.rerun(scope="fragment")
        if c2.button("👎", key=f"d{idx}"):
            _vote_and_remove(idx, 0)
            st.rerun(scope="fragment")


def _predict(w):
    """Predict scores for current images."""
    if w is None or not st.session_state[Keys.CLIP_EMBEDS]:
        return [0.0] * len(st.session_state[Keys.CLIP_IMAGES])
    X = np.array(st.session_state[Keys.CLIP_EMBEDS])
    return (X @ w).tolist()


