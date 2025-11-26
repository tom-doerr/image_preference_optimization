from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ipo.infra.constants import Keys

def _update_rows_display(st, Keys) -> None:
    try:
        from ipo.core.persistence import get_dataset_for_prompt_or_session
        X, y = get_dataset_for_prompt_or_session(st.session_state.get('prompt', ''), st.session_state)
        st.session_state[Keys.ROWS_DISPLAY] = str(X.shape[0] if X is not None else 0)
    except Exception:
        pass

__all__ = [
    "_lstate_and_prompt",
    "_sample_around_prompt",
    "_prepare_xgb_scorer",
    "_sample_one_for_batch",
    "_curation_params",
    "_curation_init_batch",
    "_curation_new_batch",
    "_curation_replace_at",
    "_curation_add",
    "_curation_train_and_next",
    "_refit_from_dataset_keep_batch",
    "_render_batch_ui",
    "run_batch_mode",
]


def _lstate_and_prompt() -> Tuple[Any, str]:
    import streamlit as st

    lstate = getattr(st.session_state, "lstate", None)
    if lstate is None:
        try:
            from ipo.core.latent_state import init_latent_state as _init

            lstate = _init()
            st.session_state.lstate = lstate
        except Exception:
            pass
    prompt = getattr(st.session_state, "prompt", None)
    if not prompt:
        from ipo.infra.constants import DEFAULT_PROMPT

        prompt = DEFAULT_PROMPT
    return lstate, prompt








def _render_tiles_row(st, idxs, lstate, prompt, steps, guidance_eff, cur_batch):
    cols = getattr(st, "columns", lambda x: [None] * x)(len(idxs))
    for col, i in zip(cols, idxs):
        args = (int(i), lstate, prompt, int(steps), float(guidance_eff), cur_batch)
        if col is not None:
            with col:
                _render_batch_tile_body(*args)
        else:
            _render_batch_tile_body(*args)


def _sample_around_prompt(scale: float = 0.8) -> np.ndarray:
    """Deterministic, stub-friendly sampler around the prompt anchor.

    Falls back to a zero anchor and a fixed RNG seed when latent_logic or RNG
    are unavailable, so tests always get a usable cur_batch without decodes.
    """
    lstate, prompt = _lstate_and_prompt()
    try:
        from ipo.core.latent_logic import z_from_prompt

        z_p = z_from_prompt(lstate, prompt)
    except Exception:
        import numpy as _np

        d = int(getattr(lstate, "d", 8))
        z_p = _np.zeros(d, dtype=float)
    rng = getattr(lstate, "rng", None)
    if rng is None:
        import numpy as _np

        rng = _np.random.default_rng(0)
        setattr(lstate, "rng", rng)
    r = rng.standard_normal(int(getattr(lstate, "d", len(z_p))))
    r = r / (np.linalg.norm(r) + 1e-12)
    z = z_p + float(getattr(lstate, "sigma", 1.0)) * float(scale) * r
    return z


def _prepare_xgb_scorer(lstate: Any, prompt: str):
    """Return (scorer, status) for XGB from cache (no auto-fit)."""
    # Prefer unified scorer; provide a tiny compat shim when tests stub only the old API.
    try:
        from ipo.core.value_scorer import get_value_scorer as _gvs
    except Exception:
        from ipo.core.value_scorer import get_value_scorer_with_status as _gvs_ws  # type: ignore

        def _gvs(vm_choice, lstate, prompt, session_state):  # type: ignore
            s, status = _gvs_ws(vm_choice, lstate, prompt, session_state)
            return (s, ("ok" if callable(s) else status))

    try:
        scorer, tag_or_status = _gvs(
            "XGBoost", lstate, prompt, __import__("streamlit").session_state
        )
        return (scorer, "ok") if scorer is not None else (None, str(tag_or_status))
    except Exception:
        return None, "xgb_unavailable"


def _sample_one_for_batch(
    lstate: Any, prompt: str, use_xgb: bool, scorer, steps: int, lr_mu: float, trust_r
) -> np.ndarray:
    """Produce one latent for the batch (hill climb removed)."""
    return _sample_around_prompt(scale=0.8)


def _curation_params():
    import streamlit as st
    vm = str(st.session_state.get(Keys.VM_CHOICE) or "")
    use_xgb = vm == "XGBoost"
    return (vm, use_xgb, 10, 0.3, None)


def _curation_init_batch() -> None:
    # Always create a fresh batch on init so each page reload/new round uses
    # newly sampled latents instead of reusing the previous cur_batch.
    _curation_new_batch()


def _curation_new_batch() -> None:
    import streamlit as st

    lstate, prompt = _lstate_and_prompt()
    z_list = []
    batch_n = int(st.session_state.get("batch_size", 6))
    # Optional XGBoost-guided hill climb per image when XGB is active.
    vm_choice, use_xgb, steps, lr_mu, trust_r = _curation_params()
    scorer = None
    scorer_status = None
    if use_xgb:
        scorer, scorer_status = _prepare_xgb_scorer(lstate, prompt)
        if scorer_status != "ok":
            scorer = None
    for i in range(batch_n):
        z_list.append(
            _sample_one_for_batch(
                lstate, prompt, use_xgb, scorer, steps, lr_mu, trust_r
            )
        )
    st.session_state.cur_batch = z_list
    st.session_state.cur_labels = [None] * len(z_list)
    st.session_state["cur_batch_nonce"] = int(st.session_state.get("cur_batch_nonce", 0)) + 1


def _curation_replace_at(idx: int) -> None:
    """Deterministically resample the item at idx; keep batch size constant.

    Under full app runs we can refresh the entire batch; in stubs/tests we
    resample only the requested index to keep behavior predictable.
    """
    import streamlit as st

    zs = list(getattr(st.session_state, "cur_batch", []) or [])
    if not zs:
        _curation_new_batch()
        zs = list(getattr(st.session_state, "cur_batch", []) or [])
        if not zs:
            return
    i = int(idx) % len(zs)
    print(f"[batch] replace_at idx={i}")
    zi = _resample_tile_at_index(i)
    zs[i] = zi
    st.session_state.cur_batch = zs


def _record_last_action_and_step(st, Keys, lstate, msg: str) -> None:
    try:
        import time as _time
        st.session_state[Keys.LAST_ACTION_TEXT] = msg
        st.session_state[Keys.LAST_ACTION_TS] = float(_time.time())
    except Exception:
        pass
    try:
        setattr(lstate, "step", int(getattr(lstate, "step", 0)) + 1)
    except Exception:
        pass


def _resample_tile_at_index(i: int) -> np.ndarray:
    """Return a deterministic resample for tile i based on batch nonce and prompt.

    Falls back to around-prompt sampling when latent logic is unavailable.
    """
    import numpy as _np
    import streamlit as st
    try:
        lstate, prompt = _lstate_and_prompt()
        try:
            from ipo.core.latent_logic import z_from_prompt as _zfp
            z_p = _zfp(lstate, prompt)
        except Exception:
            z_p = _np.zeros(int(getattr(lstate, 'd', 8)), dtype=float)
        nonce = int(st.session_state.get('cur_batch_nonce', 0))
        rng = _np.random.default_rng(1009 * (nonce + 1) + int(i) + 1)
        r = rng.standard_normal(z_p.shape)
        r = r / (float(_np.linalg.norm(r)) + 1e-12)
        return z_p + float(getattr(lstate, 'sigma', 1.0)) * 0.8 * r
    except Exception:
        return _sample_around_prompt(scale=0.8)


def _cooldown_recent(st) -> bool:
    return False  # cooldown disabled


def _fit_ridge_once(lstate, X, y, lam_now, st) -> None:
    try:
        from ipo.core.value_model import fit_value_model as _fit_vm
        _fit_vm("Ridge", lstate, X, y, float(lam_now), st.session_state)
    except Exception:
        pass


def _curation_add(label: int, z: np.ndarray, img=None) -> None:
    import streamlit as st

    from ipo.infra.constants import Keys
    from ipo.core.latent_logic import z_from_prompt

    lstate, prompt = _lstate_and_prompt()
    z_p = z_from_prompt(lstate, prompt)
    feat = (z - z_p).reshape(1, -1)
    try:
        print(f"[data] append label={int(label)}")
    except Exception:
        pass
    try:
        from ipo.core.persistence import append_sample
        append_sample(prompt, feat, float(label), img)
        _record_last_action_and_step(st, Keys, lstate, f"Labeled {'+1' if label > 0 else '-1'}")
        _update_rows_display(st, Keys)
    except Exception as e:
        print(f"[data] save failed: {e}")


def _render_good_bad(st, i, z_i, img_i):
    c1, c2 = st.columns(2)
    n = st.session_state.get("cur_batch_nonce", 0)
    with c1:
        if st.button(f"Good {i}", key=f"g_{n}_{i}"):
            _curation_add(1, z_i, img_i)
            _curation_replace_at(i)
    with c2:
        if st.button(f"Bad {i}", key=f"b_{n}_{i}"):
            _curation_add(-1, z_i, img_i)
            _curation_replace_at(i)

def _render_batch_tile_body(i: int, lstate: Any, prompt: str, steps: int, guidance_eff: float, cur_batch) -> None:
    import streamlit as st
    z_i = cur_batch[i]
    img_i = _decode_one(i, lstate, prompt, z_i, steps, guidance_eff)
    st.image(img_i, caption=f"Item {i}")
    _render_good_bad(st, i, z_i, img_i)


def _curation_train_and_next() -> None:
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    if not bool(st.session_state.get("train_on_new_data", True)):
        _curation_new_batch()
        return
    # Resolve dataset and maybe train once
    from ipo.core.persistence import get_dataset_for_prompt_or_session as _gdf
    X, y = _gdf(prompt, st.session_state)
    _maybe_train_ridge_sync(st, lstate, X, y)
    _curation_new_batch()


def _refit_from_dataset_keep_batch() -> None:
    import streamlit as st
    lstate, prompt = _lstate_and_prompt()
    if not bool(st.session_state.get("train_on_new_data", True)):
        return
    from ipo.core.persistence import get_dataset_for_prompt_or_session as _gdf
    X, y = _gdf(prompt, st.session_state)
    _maybe_train_ridge_sync(st, lstate, X, y)

def _maybe_train_ridge_sync(st, lstate, X, y) -> None:
    """Train Ridge synchronously once when data are present and cooldown allows."""
    try:
        n = int(getattr(X, "shape", (0,))[0]) if X is not None else 0
        if n <= 0 or y is None:
            return
        lam_now = float(getattr(st.session_state, Keys.REG_LAMBDA, 1e300))
        getattr(st, "toast", lambda *a, **k: None)("Training Ridge…")
        if not _cooldown_recent(st):
            _fit_ridge_once(lstate, X, y, lam_now, st)
    except Exception:
        pass


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
    from ipo.core.latent_logic import z_to_latents
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
