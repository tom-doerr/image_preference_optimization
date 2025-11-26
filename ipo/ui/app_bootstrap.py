import hashlib
import os

import streamlit as st

from ipo.infra.constants import DEFAULT_PROMPT, Keys


def init_page_and_logging() -> None:
    st.set_page_config(page_title="Latent Preference Optimizer", layout="wide")


def _early_vm_choice() -> str:
    vm = st.session_state.get(Keys.VM_CHOICE) or st.session_state.get("vm_choice") or "XGBoost"
    if not st.session_state.get(Keys.VM_CHOICE):
        st.session_state[Keys.VM_CHOICE] = vm
    return str(vm)


def _ensure_iter_defaults() -> None:
    try:
        if Keys.ITER_ETA not in st.session_state:
            st.session_state[Keys.ITER_ETA] = 0.00001
        if Keys.ITER_STEPS not in st.session_state:
            from ipo.infra.constants import DEFAULT_ITER_STEPS as _DEF
            st.session_state[Keys.ITER_STEPS] = int(_DEF)
    except Exception:
        pass


def emit_early_sidebar() -> None:
    """Keep early defaults, but do not emit sidebar lines.

    This removes the initial block (Log file / Value model / Step scores / Train/CV / Latent dim).
    """
    _early_vm_choice()
    _ensure_iter_defaults()


def ensure_prompt_and_state() -> str:
    # Prompt
    _sb_txt = getattr(st.sidebar, "text_input", st.text_input)
    base_prompt = _sb_txt("Prompt", value=(st.session_state.get("prompt") or DEFAULT_PROMPT))
    if base_prompt != st.session_state.get("prompt"):
        st.session_state.prompt = base_prompt
    # State path
    _resolve_state_path()
    # Latent state
    if "lstate" not in st.session_state or base_prompt != st.session_state.get("prompt"):
        _apply_or_init_state()
        # Initialize placeholders without decoding at import time.
        if Keys.IMAGES not in st.session_state:
            st.session_state[Keys.IMAGES] = (None, None)
    return base_prompt


def _resolve_state_path() -> None:
    sp = st.session_state.get("state_path")
    if isinstance(sp, str) and sp:
        return
    try:
        from ipo.core.persistence import state_path_for_prompt as _spp

        st.session_state.state_path = _spp(st.session_state.get("prompt") or DEFAULT_PROMPT)
    except Exception:
        h = hashlib.sha1(
            (st.session_state.get("prompt") or DEFAULT_PROMPT).encode("utf-8")
        ).hexdigest()[:10]
        st.session_state.state_path = f"latent_state_{h}.npz"


def _apply_or_init_state() -> None:
    from ipo.core.latent_state import init_latent_state, load_state

    from .app_api import _apply_state

    path = st.session_state.get("state_path")
    try:
        if isinstance(path, str) and os.path.exists(path):
            _apply_state(st, load_state(path))
            return
    except Exception:
        pass
    _apply_state(st, init_latent_state())
