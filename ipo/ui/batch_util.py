from __future__ import annotations

from typing import Any


def ensure_model_ready() -> None:
    """Ensure a decode model is loaded before any image generation."""
    try:
        from flux_local import CURRENT_MODEL_ID, set_model  # type: ignore
        if CURRENT_MODEL_ID is None:
            from ipo.infra.constants import DEFAULT_MODEL
            set_model(DEFAULT_MODEL)
    except Exception:
        pass


def prep_render_counters(st: Any) -> None:
    """Bump simple counters/nonces used to keep Streamlit keys stable."""
    try:
        st_globals = globals()
        st_globals["GLOBAL_RENDER_COUNTER"] = int(st_globals.get("GLOBAL_RENDER_COUNTER", 0)) + 1
    except Exception:
        globals()["GLOBAL_RENDER_COUNTER"] = 1
    try:
        st.session_state["render_count"] = int(st.session_state.get("render_count", 0)) + 1
    except Exception:
        pass
    try:
        st.session_state["render_nonce"] = int(st.session_state.get("render_nonce", 0)) + 1
        try:
            import secrets as __sec
            st.session_state["render_salt"] = int(__sec.randbits(32))
        except Exception:
            import time as __t
            st.session_state["render_salt"] = int(__t.time() * 1e9)
    except Exception:
        pass

