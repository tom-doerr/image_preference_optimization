import hashlib
import os
import streamlit as st

from constants import DEFAULT_PROMPT, Keys

# Early bootstrap
from app_bootstrap import init_page_and_logging, emit_early_sidebar, ensure_prompt_and_state

# App API shims (keep names stable for tests)
from ipo.ui.app_api import (
    build_controls as _build_controls,
    generate_pair as _generate_pair,
    _apply_state as _apply_state,
    _render_batch_ui as _render_batch_ui_impl,
    _curation_init_batch as _curation_init_batch_impl,
    _curation_new_batch as _curation_new_batch_impl,
    _curation_replace_at as _curation_replace_at_impl,
    _curation_add as _curation_add_impl,
    _curation_train_and_next as _curation_train_and_next_impl,
    run_app as _run_app_impl,
)
from ipo.ui.app_api import render_sidebar_tail as render_sidebar_tail_module

# State helpers (re-export minimal surface expected by tests)
def init_latent_state(*a, **k):
    from latent_state import init_latent_state as _f
    return _f(*a, **k)

def dumps_state(state):
    from latent_state import dumps_state as _f
    return _f(state)

def loads_state(data: bytes):
    from latent_state import loads_state as _f
    return _f(data)

def _export_state_bytes(state, prompt: str):
    from persistence import export_state_bytes
    return export_state_bytes(state, prompt)

init_page_and_logging()
emit_early_sidebar()

# Session defaults
if "prompt" not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT

base_prompt = ensure_prompt_and_state()
st_rerun = getattr(st, "rerun", getattr(st, "experimental_rerun", None))

# Controls
lstate = st.session_state.lstate
vm_choice, selected_gen_mode, selected_model, width, height, steps, guidance, reg_lambda, iter_steps, iter_eta, _async = _build_controls(
    st, lstate, base_prompt
)

# Sidebar tail (always render)
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


def _render_batch_ui() -> None:
    return _render_batch_ui_impl()


def _curation_init_batch() -> None:
    return _curation_init_batch_impl()


def _curation_new_batch() -> None:
    return _curation_new_batch_impl()


def _curation_replace_at(idx: int) -> None:
    return _curation_replace_at_impl(idx)


def _curation_add(label: int, z, img=None) -> None:
    return _curation_add_impl(label, z, img)


def _curation_train_and_next() -> None:
    return _curation_train_and_next_impl()


def run_app(_st, _vm_choice: str, _selected_gen_mode: str | None, _async_queue_mode: bool) -> None:
    return _run_app_impl(_st, _vm_choice, _selected_gen_mode, _async_queue_mode)


# Drive the app (batch-only)
print("[mode] dispatch async_queue_mode=False (queue removed)")
run_app(st, vm_choice, selected_gen_mode, False)

st.write(f"Interactions: {lstate.step}")
from latent_state import save_state  # local import to reduce global surface
if st.button("Reset", type="secondary"):
    _apply_state(st, init_latent_state(width=int(width), height=int(height)))
    save_state(st.session_state.lstate, st.session_state.state_path)
    if callable(st_rerun):
        st_rerun()

st.caption(
    f"Persistence: {st.session_state.state_path}{' (loaded)' if os.path.exists(st.session_state.state_path) else ''}"
)
recent = st.session_state.get(Keys.RECENT_PROMPTS, [])
if recent:
    items = [f"{hashlib.sha1(p.encode('utf-8')).hexdigest()[:10]} • {p[:30]}" for p in recent[:3]]
    st.caption("Recent states: " + ", ".join(items))
