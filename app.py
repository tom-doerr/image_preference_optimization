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
from app_api import render_sidebar_tail as render_sidebar_tail_module
from helpers import enable_file_logging
from app_bootstrap import init_page_and_logging, emit_early_sidebar, ensure_prompt_and_state
from app_api import build_controls as _build_controls
from app_api import generate_pair as _generate_pair
from app_api import _apply_state as _apply_state
from app_api import (
    _render_batch_ui as _render_batch_ui_impl,
    _curation_init_batch as _curation_init_batch_impl,
    _curation_new_batch as _curation_new_batch_impl,
    _curation_replace_at as _curation_replace_at_impl,
    _curation_add as _curation_add_impl,
    _curation_train_and_next as _curation_train_and_next_impl,
    run_app as _run_app_impl,
)

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


"""
Back-compat for tests: keep names on app module, delegated to app_api.
"""

 
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

def _render_batch_ui() -> None: return _render_batch_ui_impl()


 
def _curation_init_batch() -> None: return _curation_init_batch_impl()


def _curation_new_batch() -> None: return _curation_new_batch_impl()


def _curation_replace_at(idx: int) -> None: return _curation_replace_at_impl(idx)


def _curation_add(label: int, z, img=None) -> None: return _curation_add_impl(label, z, img)

def _curation_train_and_next() -> None: return _curation_train_and_next_impl()


 

def run_app(_st, _vm_choice: str, _selected_gen_mode: str | None, _async_queue_mode: bool) -> None: return _run_app_impl(_st, _vm_choice, _selected_gen_mode, _async_queue_mode)


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
