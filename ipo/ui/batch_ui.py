from typing import Any

import numpy as np

from ipo.infra.constants import Keys


def _update_rows_display(st, Keys):
    try:
        from ipo.core.persistence import get_dataset_for_prompt_or_session
        X, y = get_dataset_for_prompt_or_session(st.session_state.get('prompt', ''), st.session_state)
        st.session_state[Keys.ROWS_DISPLAY] = str(X.shape[0] if X is not None else 0)
    except Exception:
        pass

def _lstate_and_prompt():
    import streamlit as st

    from ipo.infra.constants import DEFAULT_PROMPT
    lstate = getattr(st.session_state, "lstate", None)
    if lstate is None:
        from ipo.core.latent_state import init_latent_state; lstate = init_latent_state(); st.session_state.lstate = lstate
    return lstate, st.session_state.get("prompt") or DEFAULT_PROMPT


def _render_tiles_row(st, idxs, lstate, prompt, steps, guidance_eff, cur_batch):
    cols = getattr(st, "columns", lambda x: [None] * x)(len(idxs))
    for col, i in zip(cols, idxs):
        args = (int(i), lstate, prompt, int(steps), float(guidance_eff), cur_batch)
        if col is not None:
            with col:
                _render_batch_tile_body(*args)
        else:
            _render_batch_tile_body(*args)


def _optimize_z(z, lstate, ss, steps, eta=0.01):
    """Gradient ascent on value function."""
    if steps <= 0: return z
    w = getattr(lstate, "w", None)
    if w is None or np.allclose(w, 0): return z
    w_norm = np.linalg.norm(w) + 1e-12
    for i in range(int(steps)):
        if i % max(1, steps // 5) == 0:
            print(f"[optim] step {i}: score={float(np.dot(w, z)):.4f}")
        z = z + eta * w / w_norm
    print(f"[optim] final: score={float(np.dot(w, z)):.4f}")
    return z

def _sample_z(lstate, prompt, scale=0.8):
    from ipo.core.latent_state import z_from_prompt
    z_p = z_from_prompt(lstate, prompt)
    rng = getattr(lstate, "rng", None) or np.random.default_rng(0)
    r = rng.standard_normal(lstate.d); r = r / (np.linalg.norm(r) + 1e-12)
    return z_p + float(lstate.sigma) * scale * r


def _curation_init_batch(): _curation_new_batch()

def _curation_new_batch() -> None:
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    n = int(st.session_state.get(Keys.BATCH_SIZE) or 3)
    steps = int(st.session_state.get(Keys.ITER_STEPS) or 0)
    eta = float(st.session_state.get(Keys.ITER_ETA) or 0.01)
    zs = [_optimize_z(_sample_z(lstate, prompt), lstate, st.session_state, steps, eta)
          for _ in range(n)]
    st.session_state.cur_batch = zs
    st.session_state.cur_labels = [None] * n
    st.session_state["cur_batch_nonce"] = int(st.session_state.get("cur_batch_nonce", 0)) + 1


def _curation_replace_at(idx: int) -> None:
    import streamlit as st
    zs = list(getattr(st.session_state, "cur_batch", []) or [])
    if not zs: _curation_new_batch(); return
    lstate, prompt = _lstate_and_prompt()
    steps = int(st.session_state.get(Keys.ITER_STEPS) or 0)
    eta = float(st.session_state.get(Keys.ITER_ETA) or 0.01)
    z_new = _optimize_z(_sample_z(lstate, prompt), lstate, st.session_state, steps, eta)
    zs[int(idx) % len(zs)] = z_new
    st.session_state.cur_batch = zs


def _curation_add(label: int, z: np.ndarray, img=None) -> None:
    import streamlit as st

    from ipo.core.latent_state import z_from_prompt
    from ipo.core.persistence import append_sample
    lstate, prompt = _lstate_and_prompt()
    feat = (z - z_from_prompt(lstate, prompt)).reshape(1, -1)
    print(f"[add] label={label} feat.shape={feat.shape} prompt={prompt[:20]}...")
    append_sample(prompt, feat, float(label), img)
    lstate.step = getattr(lstate, "step", 0) + 1
    print(f"[label] {'Good' if label > 0 else 'Bad'} (step {lstate.step})")
    _update_rows_display(st, Keys)
    # Train after each label
    if st.session_state.get("train_on_new_data", True):
        _train_if_data(st, lstate, prompt)


def _render_good_bad(st, i, z_i, img_i):
    c1, c2 = st.columns(2)
    n = st.session_state.get("cur_batch_nonce", 0)
    with c1:
        if st.button(f"Good {i}", key=f"g_{n}_{i}"):
            print(f"[btn] Good {i} clicked")
            _curation_add(1, z_i, img_i)
            _curation_replace_at(i)
    with c2:
        if st.button(f"Bad {i}", key=f"b_{n}_{i}"):
            print(f"[btn] Bad {i} clicked")
            _curation_add(-1, z_i, img_i)
            _curation_replace_at(i)

def _render_batch_tile_body(i: int, lstate: Any, prompt: str, steps: int, guidance_eff: float, cur_batch) -> None:
    import streamlit as st
    z_i = cur_batch[i]
    img_i = _decode_one(i, lstate, prompt, z_i, steps, guidance_eff)
    st.image(img_i, caption=f"Item {i}")
    _render_good_bad(st, i, z_i, img_i)

def _train_if_data(st, lstate, prompt):
    from ipo.core.persistence import get_dataset_for_prompt_or_session as gd
    from ipo.core.value_model import fit_value_model as fv
    X, y = gd(prompt, st.session_state)
    vm = st.session_state.get(Keys.VM_CHOICE) or "Ridge"
    if X is not None and X.shape[0] > 0:
        n_pos, n_neg = int((y > 0).sum()), int((y < 0).sum())
        print(f"[train] {vm} on {X.shape[0]} samples (+{n_pos} / -{n_neg})")
        fv(vm, lstate, X, y, float(getattr(st.session_state, Keys.REG_LAMBDA, 1e300)), st.session_state)

def _curation_train_and_next():
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    if st.session_state.get("train_on_new_data", True): _train_if_data(st, lstate, prompt)
    _curation_new_batch()


def _refit_from_dataset_keep_batch():
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    if st.session_state.get("train_on_new_data", True): _train_if_data(st, lstate, prompt)


def _render_batch_ui() -> None:
    import streamlit as st

    # Header + init
    lstate, prompt, steps, guidance_eff, cur_batch = _batch_init(st)
    n = len(cur_batch)
    if n == 0:
        return
    per_row = min(5, n)

    for row_start in range(0, n, per_row):
        row_end = min(row_start + per_row, n)
        _render_tiles_row(st, list(range(row_start, row_end)), lstate, prompt,
                          steps, guidance_eff, cur_batch)


def run_batch_mode() -> None:
    _curation_init_batch()
    _render_batch_ui()
def _decode_one(i: int, lstate: Any, prompt: str, z_i: np.ndarray, steps: int, guidance_eff: float):
    from ipo.core.latent_state import z_to_latents
    from ipo.infra.pipeline_local import generate_flux_image_latents
    la = z_to_latents(lstate, z_i)
    return generate_flux_image_latents(prompt, latents=la, width=lstate.width, height=lstate.height, steps=steps, guidance=guidance_eff)


def _batch_init(st):
    from ipo.infra.pipeline_local import set_model
    set_model(None)  # ensure model ready
    (getattr(st, "subheader", lambda *a, **k: None))("Curation batch")
    lstate, prompt = _lstate_and_prompt()
    steps = int(getattr(st.session_state, "steps", 6) or 6)
    guidance_eff = float(getattr(st.session_state, "guidance_eff", 0.0) or 0.0)
    cur_batch = getattr(st.session_state, "cur_batch", []) or []
    if not cur_batch:
        _curation_init_batch()
        cur_batch = getattr(st.session_state, "cur_batch", []) or []
    return lstate, prompt, steps, guidance_eff, cur_batch
