import hashlib
import os
import types
import streamlit as st
from ipo.infra.constants import DEFAULT_PROMPT, Keys
from ipo.infra.util import enable_file_logging, safe_write


def init_page_and_logging() -> None:
    st.set_page_config(page_title="Latent Preference Optimizer", layout="wide")
    try:
        log_path = enable_file_logging()
        st.sidebar.write(f"Log file: {log_path}")
    except Exception:
        pass


def emit_early_sidebar() -> None:
    vm = st.session_state.get(Keys.VM_CHOICE) or st.session_state.get("vm_choice") or "XGBoost"
    if not st.session_state.get(Keys.VM_CHOICE):
        st.session_state[Keys.VM_CHOICE] = vm
    safe_write(st, f"Value model: {vm}")
    safe_write(st, "Step scores: n/a")
    # Ensure iterative params exist early so tests can observe them on import
    try:
        if Keys.ITER_ETA not in st.session_state:
            st.session_state[Keys.ITER_ETA] = 0.00001
        if Keys.ITER_STEPS not in st.session_state:
            from ipo.infra.constants import DEFAULT_ITER_STEPS as _DEF
            st.session_state[Keys.ITER_STEPS] = int(_DEF)
    except Exception:
        pass
    try:
        from value_scorer import get_value_scorer as _gvs

        lstate_early = getattr(st.session_state, "lstate", None) or types.SimpleNamespace(d=0, w=None)  # type: ignore
        prompt_early = (
            st.session_state.get(Keys.PROMPT) or st.session_state.get("prompt") or DEFAULT_PROMPT
        )
        scorer, tag_or_status = _gvs(vm, lstate_early, prompt_early, st.session_state)
        vs_status_early = "ok" if scorer is not None else str(tag_or_status)
        active_early = "yes" if (vm == "XGBoost" and scorer is not None) else "no"
    except Exception:
        vs_status_early = "xgb_unavailable" if vm == "XGBoost" else "ridge_untrained"
        active_early = "no"
    for ln in (
        "Train score: n/a",
        "CV score: n/a",
        "Last CV: n/a",
        "Last train: n/a",
        f"Value scorer status: {vs_status_early}",
        f"Value scorer: {vm} (n/a, rows=0)",
        f"XGBoost active: {active_early}",
        "Optimization: Ridge only",
    ):
        safe_write(st, ln)
    ld = int(getattr(getattr(st.session_state, "lstate", None), "d", 0)) if hasattr(st, "session_state") else 0
    safe_write(st, f"Latent dim: {ld}")
    try:
        from flux_local import set_model

        set_model("stabilityai/sd-turbo")
    except Exception:
        pass


def ensure_prompt_and_state() -> str:
    # Prompt
    _sb_txt = getattr(st.sidebar, "text_input", st.text_input)
    base_prompt = _sb_txt("Prompt", value=(st.session_state.get("prompt") or DEFAULT_PROMPT))
    if base_prompt != st.session_state.get("prompt"):
        st.session_state.prompt = base_prompt
    # State path
    sp = st.session_state.get("state_path")
    if not isinstance(sp, str) or not sp:
        try:
            from ipo.core.persistence import state_path_for_prompt as _spp

            st.session_state.state_path = _spp(st.session_state.prompt)
        except Exception:
            h = hashlib.sha1(st.session_state.prompt.encode("utf-8")).hexdigest()[:10]
            st.session_state.state_path = f"latent_state_{h}.npz"
    # Latent state
    if "lstate" not in st.session_state or base_prompt != st.session_state.get("prompt"):
        from latent_state import init_latent_state, load_state

        if os.path.exists(st.session_state.state_path):
            try:
                from .app_api import _apply_state
                _apply_state(st, load_state(st.session_state.state_path))
            except Exception:
                from .app_api import _apply_state
                _apply_state(st, init_latent_state())
        else:
            from .app_api import _apply_state
            _apply_state(st, init_latent_state())
        # Initialize placeholders without decoding at import time.
        if Keys.IMAGES not in st.session_state:
            st.session_state[Keys.IMAGES] = (None, None)
    return base_prompt
