import numpy as np
import streamlit as st

from ipo.infra.constants import (
    DEFAULT_CURATION_SIZE,
    DEFAULT_ITER_ETA,
    DEFAULT_ITER_STEPS,
    DEFAULT_PROMPT,
    DEFAULT_SERVER_URL,
    Keys,
)


def _get_steps(ss):
    return int(ss.get(Keys.ITER_STEPS) or DEFAULT_ITER_STEPS)


def _get_eta(ss):
    return float(ss.get(Keys.ITER_ETA) or DEFAULT_ITER_ETA)


def _get_per_row(ss, n):
    import math
    ipr = int(ss.get(Keys.IMAGES_PER_ROW) or -1)
    return ipr if ipr > 0 else max(1, int(math.ceil(math.sqrt(n))))


def _update_rows_display(st, Keys):
    X = st.session_state.get(Keys.DATASET_X)
    st.session_state[Keys.ROWS_DISPLAY] = str(X.shape[0] if X is not None else 0)

def _lstate_and_prompt():
    import streamlit as st

    lstate = getattr(st.session_state, "lstate", None)
    if lstate is None:
        from ipo.core.latent_state import init_latent_state
        sm = st.session_state.get(Keys.SPACE_MODE, "Latent")
        lstate = init_latent_state(space_mode=sm)
        st.session_state.lstate = lstate
    return lstate, st.session_state.get("prompt") or DEFAULT_PROMPT


from ipo.core.latent_optimizer import optimize_latent as _optimize_z

def _get_good_mean(prompt, ss):
    from ipo.ui.sampling import get_good_mean
    X, y = ss.get(Keys.DATASET_X), ss.get(Keys.DATASET_Y)
    return get_good_mean(X, y)

def _get_good_dist(ss):
    from ipo.ui.sampling import get_good_dist
    X, y = ss.get(Keys.DATASET_X), ss.get(Keys.DATASET_Y)
    return get_good_dist(X, y)


def _random_offset(lstate, scale=0.8):
    from ipo.ui.sampling import random_offset
    rng = getattr(lstate, "rng", None) or np.random.default_rng()
    return random_offset(lstate.d, float(lstate.sigma), rng, scale)

def _sample_z(lstate, prompt, scale=0.8):
    import streamlit as st

    from ipo.core.latent_state import z_from_prompt
    z_p = z_from_prompt(lstate, prompt)
    mode = st.session_state.get(Keys.SAMPLE_MODE) or "AvgGood"
    gm = _get_good_mean(prompt, st.session_state)
    if mode == "AvgGood" and gm is not None:
        return z_p + gm
    if mode == "Prompt+AvgGood" and gm is not None:
        return z_p + gm * 0.5
    if mode == "Prompt":
        return z_p
    if mode == "GoodDist":
        mu, sigma = _get_good_dist(st.session_state)
        if mu is not None:
            rng = getattr(lstate, "rng", None) or np.random.default_rng()
            return mu + sigma * rng.standard_normal(len(mu))
    return z_p + _random_offset(lstate, scale)


def _curation_init_batch():
    _curation_new_batch()

def _curation_new_batch() -> None:
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    n = int(st.session_state.get(Keys.CURATION_SIZE) or DEFAULT_CURATION_SIZE)
    steps, eta = _get_steps(st.session_state), _get_eta(st.session_state)
    zs = [_optimize_z(_sample_z(lstate, prompt), lstate, st.session_state, steps, eta)
          for _ in range(n)]
    st.session_state.cur_batch = zs
    st.session_state.cur_labels = [None] * n
    st.session_state[Keys.CURATION_NONCE] = int(st.session_state.get(Keys.CURATION_NONCE, 0)) + 1


def _curation_replace_at(idx: int) -> None:
    import streamlit as st
    zs = list(getattr(st.session_state, "cur_batch", []) or [])
    if not zs:
        _curation_new_batch()
        return
    lstate, prompt = _lstate_and_prompt()
    steps, eta = _get_steps(st.session_state), _get_eta(st.session_state)
    z_new = _optimize_z(_sample_z(lstate, prompt), lstate, st.session_state, steps, eta)
    zs[int(idx) % len(zs)] = z_new
    st.session_state.cur_batch = zs


def _curation_add(label: int, z: np.ndarray, img=None, skip_train=False) -> None:
    import streamlit as st

    from ipo.core.latent_state import z_from_prompt
    from ipo.core.persistence import append_sample
    lstate, prompt = _lstate_and_prompt()
    feat = (z - z_from_prompt(lstate, prompt)).reshape(1, -1)
    append_sample(prompt, feat, float(label), img)
    lstate.step = getattr(lstate, "step", 0) + 1
    if not skip_train:
        _update_rows_display(st, Keys)
        if st.session_state.get("train_on_new_data", True):
            _train_if_data(st, lstate, prompt)


def _train_if_data(st, lstate, prompt):
    from ipo.core.value_model import fit_value_model as fv
    X = st.session_state.get(Keys.DATASET_X)
    y = st.session_state.get(Keys.DATASET_Y)
    vm = st.session_state.get(Keys.VM_CHOICE) or "Ridge"
    if X is not None and X.shape[0] > 0:
        n_pos, n_neg = int((y > 0).sum()), int((y < 0).sum())
        print(f"[train] {vm} on {X.shape[0]} samples (+{n_pos} / -{n_neg})")
        with st.spinner(f"Training {vm}..."):
            fv(vm, lstate, X, y, float(st.session_state.get(Keys.REG_LAMBDA) or 1000), st.session_state)

def _curation_train_and_next():
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    if st.session_state.get("train_on_new_data", True):
        _train_if_data(st, lstate, prompt)
    _curation_new_batch()


def _refit_from_dataset_keep_batch():
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    if st.session_state.get("train_on_new_data", True):
        _train_if_data(st, lstate, prompt)


def run_batch_mode():
    from ipo.core.model_manager import ModelManager
    from ipo.infra.model_registry import get_model_config
    model_name = st.session_state.get(Keys.SELECTED_MODEL) or "model"
    cfg = get_model_config(model_name) or {}
    size = cfg.get("size", "")
    prec = cfg.get("precision", "")
    info = f" ({size}, {prec})" if size else ""
    with st.spinner(f"Loading {model_name}{info}..."):
        ModelManager.ensure_ready()
    lstate, prompt = _lstate_and_prompt()
    n = int(st.session_state.get(Keys.CURATION_SIZE) or DEFAULT_CURATION_SIZE)
    if "batch_z" not in st.session_state or len(st.session_state.batch_z) != n:
        st.session_state.batch_z = [None] * n
        st.session_state.batch_img = [None] * n
    gen_count = sum(1 for img in st.session_state.batch_img if img is not None)
    counter = st.sidebar.empty()
    counter.text(f"Generated: {gen_count}/{n}")
    _render_batch(lstate, prompt, n, counter)

def _render_batch(lstate, prompt, n, counter=None):
    from ipo.core.latent_state import z_to_latents
    from ipo.infra.pipeline_local import generate_flux_image_latents as gen
    if st.session_state.get(Keys.CURATION_FORM_MODE):
        _render_batch_form(lstate, prompt, n, z_to_latents, gen, counter)
    else:
        _render_batch_buttons(lstate, prompt, n, z_to_latents, gen, counter)

def _render_batch_form(ls, pr, n, z2l, gen, counter=None):
    per_row = _get_per_row(st.session_state, n)
    checks = []
    css = "<style>div[data-testid='stCheckbox'] input {transform:scale(5)}</style>"
    st.markdown(css, unsafe_allow_html=True)
    with st.form("batch_form"):
        for row in range(0, n, per_row):
            cols = st.columns(min(per_row, n - row))
            for j, col in enumerate(cols):
                with col:
                    _render_tile_chk(row + j, ls, pr, z2l, gen, checks, n, counter)
        if st.form_submit_button("Submit"):
            _submit_batch(ls, pr, checks)

def _render_tile_chk(i, ls, pr, z2l, gen, checks, n=0, counter=None):
    _ensure_tile(i, ls, pr, z2l, gen, n, counter)
    z = st.session_state.batch_z[i]
    sc, d = _get_score(z, ls, st.session_state), _get_dist(z, ls, pr)
    st.image(st.session_state.batch_img[i], caption=f"#{i} s={sc:.3f} d={d:.2f}")
    zhash = hash(z.tobytes()) & 0xFFFFFFFF
    checks.append(st.checkbox("Good", key=f"chk{zhash}_{i}"))

def _submit_batch(ls, pr, checks):
    with st.spinner(f"Saving {len(checks)} samples..."):
        for i, good in enumerate(checks):
            label = 1 if good else -1
            z, img = st.session_state.batch_z[i], st.session_state.batch_img[i]
            _curation_add(label, z, img, skip_train=True)
    print(f"[submit] {len(checks)} samples, reloading")
    from ipo.core.persistence import get_dataset_for_prompt_or_session
    X, y = get_dataset_for_prompt_or_session(pr, st.session_state)
    st.session_state[Keys.DATASET_X] = X
    st.session_state[Keys.DATASET_Y] = y
    _train_if_data(st, ls, pr)
    _update_rows_display(st, Keys)
    n = len(checks)
    st.session_state.batch_z = [None] * n
    st.session_state.batch_img = [None] * n
    st.rerun()

def _ensure_tile(i, ls, pr, z2l, gen, n=0, counter=None):
    print(f"[_ensure_tile] i={i}")
    if st.session_state.batch_z[i] is None:
        with st.spinner(f"Optimizing latent {i+1}/{n}..."):
            s, e = _get_steps(st.session_state), _get_eta(st.session_state)
            st.session_state.batch_z[i] = _optimize_z(_sample_z(ls, pr), ls, st.session_state, s, e)
        print(f"[_ensure_tile] i={i} z optimized")
    if st.session_state.batch_img[i] is None:
        with st.spinner(f"Generating image {i+1}/{n}..."):
            steps = int(st.session_state.get(Keys.STEPS) or 6)
            z = st.session_state.batch_z[i]
            seed = int(st.session_state.get(Keys.NOISE_SEED) or 42)
            st.session_state.batch_img[i] = _gen_img(z, ls, pr, steps, seed, z2l, gen)
        print(f"[_ensure_tile] i={i} img done")
        if counter and n > 0:
            gc = sum(1 for img in st.session_state.batch_img if img is not None)
            counter.text(f"Generated: {gc}/{n}")

def _gen_img(z, ls, pr, steps, seed, z2l, gen):
    from ipo.core.model_manager import ModelManager
    ModelManager.ensure_ready()
    if st.session_state.get(Keys.GEN_MODE) == "server":
        return _gen_img_server(z, ls, pr, steps, seed)
    return _gen_img_local(z, ls, pr, steps, seed, z2l, gen)


def _gen_img_server(z, ls, pr, steps, seed):
    from ipo.server.gen_client import GenerationClient
    url = st.session_state.get(Keys.GEN_SERVER_URL) or DEFAULT_SERVER_URL
    return GenerationClient(url).generate(pr, z.tolist(), "latent", ls.width, ls.height, steps, 0.0, seed, 1.0)


def _gen_img_local(z, ls, pr, steps, seed, z2l, gen):
    model = st.session_state.get(Keys.SELECTED_MODEL) or ""
    if "flux" in model.lower():
        from ipo.infra.pipeline_local import generate
        return generate(pr, ls.width, ls.height, steps, 0.0, int(abs(hash(z.tobytes())) % (2**31)))
    return gen(pr, z2l(ls, z), ls.width, ls.height, steps, 0.0)


def _render_batch_buttons(ls, pr, n, z2l, gen, counter=None):
    per_row = _get_per_row(st.session_state, n)
    for row in range(0, n, per_row):
        cols = st.columns(min(per_row, n - row))
        for j, col in enumerate(cols):
            with col:
                _render_tile(row + j, ls, pr, z2l, gen, n, counter)

def _render_tile(i, ls, pr, z2l, gen, n=0, counter=None):
    _ensure_tile(i, ls, pr, z2l, gen, n, counter)
    z = st.session_state.batch_z[i]
    sc = _get_score(z, ls, st.session_state)
    d = _get_dist(z, ls, pr)
    st.image(st.session_state.batch_img[i], caption=f"#{i} s={sc:.3f} d={d:.2f}")
    if st.button("👍", key=f"g{i}"):
        _do_label(i, 1, ls, pr)
    if st.button("👎", key=f"b{i}"):
        _do_label(i, -1, ls, pr)

def _do_label(i, label, ls, pr):
    _curation_add(label, st.session_state.batch_z[i], st.session_state.batch_img[i])
    if st.session_state.get(Keys.REGEN_ALL):
        n = len(st.session_state.batch_z)
        st.session_state.batch_z = [None] * n
        st.session_state.batch_img = [None] * n
    else:
        st.session_state.batch_z[i] = None
        st.session_state.batch_img[i] = None
    st.rerun()

def _get_score(z, ls, ss):
    vm = ss.get(Keys.VM_CHOICE) or "Ridge"
    if vm == "XGBoost":
        from ipo.core.value_model import _get_xgb_model, _xgb_proba
        mdl = _get_xgb_model(ss)
        return _xgb_proba(mdl, z) if mdl else 0.0
    w = getattr(ls, "w", None)
    return float(np.dot(w, z)) if w is not None else 0.0

def _get_dist(z, ls, prompt):
    from ipo.core.latent_state import z_from_prompt
    z0 = z_from_prompt(ls, prompt)
    return float(np.linalg.norm(z - z0))

