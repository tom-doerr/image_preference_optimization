import streamlit as st
import types
import concurrent.futures as futures  # re-exported for tests
import numpy as np
import os, hashlib
from constants import DEFAULT_PROMPT, Keys
import batch_ui as _batch_ui
# Prefer direct latent_state imports to avoid test stubs shadowing latent_opt
def init_latent_state(*a, **k):
    from latent_state import init_latent_state as _f; return _f(*a, **k)
def dumps_state(state):
    from latent_state import dumps_state as _f; return _f(state)
def loads_state(data: bytes):
    from latent_state import loads_state as _f; return _f(data)
from flux_local import set_model
from ui_sidebar import render_sidebar_tail as render_sidebar_tail_module
from helpers import enable_file_logging
from app_bootstrap import init_page_and_logging, emit_early_sidebar, ensure_prompt_and_state
from app_api import build_controls as _build_controls
from app_api import generate_pair as _generate_pair

# _safe_write removed (199h)
def _export_state_bytes(state, prompt: str):
    from persistence import export_state_bytes as _rpc
    return _rpc(state, prompt)

def update_latent_ridge(*a, **k):
    from latent_logic import update_latent_ridge as _f; return _f(*a, **k)  # type: ignore
def z_from_prompt(*a, **k):
    from latent_logic import z_from_prompt as _f; return _f(*a, **k)  # type: ignore
def propose_latent_pair_ridge(*a, **k):
    from latent_logic import propose_latent_pair_ridge as _f; return _f(*a, **k)  # type: ignore

 


init_page_and_logging()
st_rerun=getattr(st,"rerun",getattr(st,"experimental_rerun",None))
emit_early_sidebar()




# Back-compat for tests: keep names on app module
 

 

def _apply_state(*args) -> None:  # re-exported for tests
    # Flexible arity for test/back-compat: _apply_state(new_state) or _apply_state(st, new_state)
    if len(args) == 1:
        st_local, new_state = st, args[0]
    elif len(args) == 2:
        st_local, new_state = args  # type: ignore[misc]
    else:
        raise TypeError("_apply_state() expects 1 or 2 arguments")
    from constants import Keys as _K

    def _init_pair_for_state() -> None:
        try:
            from latent_opt import propose_next_pair; z1, z2 = propose_next_pair(new_state, st.session_state.prompt); st.session_state.lz_pair = (z1, z2); return
        except Exception:
            pass
        try:
            from latent_logic import propose_latent_pair_ridge; st.session_state.lz_pair = propose_latent_pair_ridge(new_state); return
        except Exception:
            pass
        try:
            import numpy as _np; d = int(getattr(new_state, "d", 0)); st.session_state.lz_pair = (_np.zeros(d, dtype=float), _np.zeros(d, dtype=float))
        except Exception:
            st.session_state.lz_pair = (None, None)

    def _reset_derived_state() -> None:
        import numpy as _np
        st.session_state[_K.IMAGES] = (None, None); st.session_state[_K.MU_IMAGE] = None
        if getattr(new_state, "mu", None) is None: setattr(new_state, "mu", _np.zeros(int(getattr(new_state, "d", 0)), dtype=float))
        _mh = getattr(new_state, "mu_hist", None) or []
        st.session_state.mu_history = [m.copy() for m in _mh] or [new_state.mu.copy()]
        st.session_state.mu_best_idx = 0; st.session_state.prompt_image = None
        for k in ("next_prefetch", "_bg_exec"): st.session_state.pop(k, None)

    st_local.session_state.lstate = new_state
    try:
        use_rand = bool(getattr(st_local.session_state, _K.USE_RANDOM_ANCHOR, False))
        setattr(new_state, "use_random_anchor", use_rand)
        setattr(new_state, "random_anchor_z", None)
    except Exception:
        pass
    _init_pair_for_state()
    _reset_derived_state()
    # Random μ init around the prompt anchor when μ is all zeros.
    try:
        import numpy as _np
        from latent_logic import z_from_prompt as _zfp

        if _np.allclose(new_state.mu, 0.0):
            pr = st_local.session_state.get(Keys.PROMPT) or st_local.session_state.get("prompt") or DEFAULT_PROMPT
            z_p = _zfp(new_state, pr)
            r = new_state.rng.standard_normal(new_state.d).astype(float)
            nr = float(_np.linalg.norm(r))
            if nr > 0.0:
                r = r / nr
            new_state.mu = z_p + float(new_state.sigma) * r
    except Exception:
        pass

 
if "prompt" not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT
if "xgb_train_async" not in st.session_state:
    try:
        st.session_state["xgb_train_async"] = True
    except Exception:
        try:
            setattr(st.session_state, "xgb_train_async", True)
        except Exception:
            pass
 

base_prompt = ensure_prompt_and_state()

def build_controls(st, lstate, base_prompt):  # noqa: E402
    return _build_controls(st, lstate, base_prompt)

lstate = st.session_state.lstate
z_a, z_b = st.session_state.lz_pair
vm_choice, selected_gen_mode, selected_model, width, height, steps, guidance, reg_lambda, iter_steps, iter_eta, async_queue_mode = build_controls(
    st, lstate, base_prompt
)
 
try:
    if not getattr(st.session_state, "cur_batch", None):
        _batch_ui._curation_init_batch()
except Exception:
    pass
 
if hasattr(st.session_state, "apply_size_clicked") and st.session_state.apply_size_clicked:
    st.session_state.apply_size_clicked = False


render_sidebar_tail_module(
    st,
    lstate,
    st.session_state.prompt,
    st.session_state.state_path,
    vm_choice,
    int(iter_steps),
    float(iter_eta) if iter_eta is not None else None,
    selected_model,
    _apply_state,
    st_rerun,
)


def generate_pair():
    return _generate_pair(base_prompt)

def _render_batch_ui() -> None: return _batch_ui._render_batch_ui()


 
def _curation_init_batch() -> None:
    try:
        _batch_ui._curation_init_batch()
    except Exception:
        pass
    # Ensure a minimal deterministic batch exists for stubs (no decode path)
    try:
        if not getattr(st.session_state, "cur_batch", None):
            import numpy as _np
            try:
                from latent_logic import z_from_prompt as _zfp

                z_p = _zfp(st.session_state.lstate, st.session_state.prompt)
            except Exception:
                d = int(getattr(st.session_state.lstate, "d", 8))
                z_p = _np.zeros(d, dtype=float)
            n = int(getattr(st.session_state, "batch_size", 4))
            rng = _np.random.default_rng(0)
            zs = [z_p + 0.01 * rng.standard_normal(z_p.shape) for _ in range(n)]
            st.session_state.cur_batch = zs
            st.session_state.cur_labels = [None] * n
    except Exception:
        pass


def _curation_new_batch() -> None:
    try:
        _batch_ui._curation_new_batch()
    except Exception:
        pass
    # Stub-friendly refresh if batch creation failed
    try:
        if not getattr(st.session_state, "cur_batch", None):
            _curation_init_batch()
    except Exception:
        pass


def _curation_replace_at(idx: int) -> None:
    try:
        _batch_ui._curation_replace_at(idx)
    except Exception:
        pass
    # Deterministic resample for stubs
    try:
        import numpy as _np
        zs = getattr(st.session_state, "cur_batch", None)
        if isinstance(zs, list) and len(zs) > 0:
            try:
                from latent_logic import z_from_prompt as _zfp

                z_p = _zfp(st.session_state.lstate, st.session_state.prompt)
            except Exception:
                d = int(getattr(st.session_state.lstate, "d", 8))
                z_p = _np.zeros(d, dtype=float)
            rng = _np.random.default_rng(idx + 1)
            zs[idx % len(zs)] = z_p + 0.01 * rng.standard_normal(z_p.shape)
            st.session_state.cur_batch = zs
    except Exception:
        pass


def _curation_add(label: int, z, img=None) -> None:
    try:
        return _batch_ui._curation_add(label, z, img)
    except Exception:
        return None

def _curation_train_and_next() -> None:
    try:
        return _batch_ui._curation_train_and_next()
    except Exception:
        return None


 

def run_app(_st, _vm_choice: str, _selected_gen_mode: str | None, _async_queue_mode: bool) -> None:
    _batch_ui.run_batch_mode()


try: async_queue_mode
except NameError: async_queue_mode = False
try: print("[mode] dispatch async_queue_mode=False (queue removed)"); run_app(st, vm_choice, selected_gen_mode, False)
except Exception: pass
try: 
    if not getattr(st.session_state, "cur_batch", None): _curation_init_batch()
except Exception: pass

st.write(f"Interactions: {lstate.step}")
if st.button("Reset", type="secondary"): _apply_state(st, init_latent_state(width=int(width), height=int(height))); save_state(st.session_state.lstate, st.session_state.state_path); (st_rerun() if callable(st_rerun) else None)

st.caption(f"Persistence: {st.session_state.state_path}{' (loaded)' if os.path.exists(st.session_state.state_path) else ''}")
recent = st.session_state.get(Keys.RECENT_PROMPTS, [])
if recent:
    items = [f"{hashlib.sha1(p.encode('utf-8')).hexdigest()[:10]} • {p[:30]}" for p in recent[:3]]; st.caption("Recent states: " + ", ".join(items))
