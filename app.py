import hashlib
import os

import streamlit as st

from ipo.infra.constants import DEFAULT_PROMPT, Keys
from ipo.ui.app_api import (
    _apply_state as _apply_state,
)
from ipo.ui.app_api import (
    _curation_add as _curation_add_impl,
)
from ipo.ui.app_api import (
    _curation_init_batch as _curation_init_batch_impl,
)
from ipo.ui.app_api import (
    _curation_new_batch as _curation_new_batch_impl,
)
from ipo.ui.app_api import (
    _curation_replace_at as _curation_replace_at_impl,
)
from ipo.ui.app_api import (
    _curation_train_and_next as _curation_train_and_next_impl,
)
from ipo.ui.app_api import (
    _render_batch_ui as _render_batch_ui_impl,
)

# App API shims (keep names stable for tests)
from ipo.ui.app_api import (
    build_controls as _build_controls,
)
from ipo.ui.app_api import (
    generate_pair as _generate_pair,
)
from ipo.ui.app_api import render_sidebar_tail as render_sidebar_tail_module
from ipo.ui.app_api import (
    run_app as _run_app_impl,
)

# Early bootstrap
from ipo.ui.app_bootstrap import (
    emit_early_sidebar,
    ensure_prompt_and_state,
    init_page_and_logging,
)


# State helpers (re-export minimal surface expected by tests)
def init_latent_state(*a, **k):
    from ipo.core.latent_state import init_latent_state as _f
    return _f(*a, **k)

def dumps_state(state):
    from ipo.core.latent_state import dumps_state as _f
    return _f(state)

def loads_state(data: bytes):
    from ipo.core.latent_state import loads_state as _f
    return _f(data)

def _export_state_bytes(state, prompt: str):
    from ipo.core.persistence import export_state_bytes
    return export_state_bytes(state, prompt)

init_page_and_logging()
emit_early_sidebar()

# Session defaults
if "prompt" not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT

base_prompt = ensure_prompt_and_state()
st_rerun = getattr(st, "rerun", getattr(st, "experimental_rerun", None))

# Keep user-selectable value models (XGBoost, Logistic, Ridge). No forced override.

# Controls
lstate = st.session_state.lstate
(
    vm_choice,
    selected_gen_mode,
    selected_model,
    width,
    height,
    steps,
    guidance,
    reg_lambda,
    iter_steps,
    iter_eta,
    _,
) = _build_controls(st, lstate, base_prompt)

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


def run_app(_st, _vm_choice: str, _selected_gen_mode: str | None) -> None:
    return _run_app_impl(_st, _vm_choice, _selected_gen_mode)


# Drive the app (batch-only, sync-only)
run_app(st, vm_choice, selected_gen_mode)

st.write(f"Interactions: {getattr(lstate, 'step', 0)}")
from ipo.core.latent_state import save_state  # noqa: E402  (local import to reduce global surface)

if st.button("Reset", type="secondary"):
    _apply_state(st, init_latent_state(width=int(width), height=int(height)))
    save_state(st.session_state.lstate, st.session_state.state_path)
    if callable(st_rerun):
        st_rerun()

_sp = st.session_state.get("state_path")
if isinstance(_sp, str) and _sp:
    st.caption(f"Persistence: {_sp}{' (loaded)' if os.path.exists(_sp) else ''}")
recent = st.session_state.get(Keys.RECENT_PROMPTS, [])
if recent:
    items = [f"{hashlib.sha1(p.encode('utf-8')).hexdigest()[:10]} • {p[:30]}" for p in recent[:3]]
    st.caption("Recent states: " + ", ".join(items))
