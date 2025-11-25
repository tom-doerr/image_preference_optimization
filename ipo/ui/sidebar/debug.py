"""Debug/log-tail and last-call panels for the sidebar.

Separated from `ui_sidebar` to keep that module lean while preserving
behavior and strings. All functions here are imported and used by
`ipo.ui.ui_sidebar` without changes to call sites.
"""
from __future__ import annotations

from typing import Any

from ipo.infra.util import safe_write


def _lc_write_key(st: Any, lc: dict, key: str) -> None:
    try:
        if key in lc:
            safe_write(st, f"{key}: {lc[key]}")
    except Exception:
        pass


def _lc_warn_std(st: Any, lc: dict) -> None:
    try:
        stdv = lc.get("latents_std")
        if stdv is not None and float(stdv) <= 1e-9:
            st.sidebar.write(f"warn: latents std {float(stdv):.3g}")
    except Exception:
        pass


def _emit_last_call_info(st: Any) -> None:
    try:
        from ipo.infra.pipeline_local import get_last_call  # type: ignore

        lc = get_last_call() or {}
    except Exception:
        lc = {}
    for k in ("model_id", "event", "width", "height", "latents_std", "latents_mean"):
        _lc_write_key(st, lc, k)
    _lc_warn_std(st, lc)
    try:
        w = lc.get("width")
        h = lc.get("height")
        if w is not None and h is not None:
            st.sidebar.write(f"pipe_size: {w}x{h}")
    except Exception:
        pass


def _emit_log_tail(st: Any) -> None:
    try:
        import logging as _logging

        _logging.getLogger("ipo").setLevel(_logging.DEBUG)
        n_default = int(st.session_state.get("debug_tail_lines", 30) or 30)
        n_lines = int(getattr(st.sidebar, 'number_input', lambda *a, **k: n_default)(
            'Debug log tail (lines)', value=n_default, step=10
        ) or n_default)
        st.session_state["debug_tail_lines"] = n_lines
        try:
            with open('ipo.debug.log', 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[-int(max(1, n_lines)) :]
            if hasattr(st.sidebar, 'expander') and callable(getattr(st.sidebar, 'expander', None)):
                with st.sidebar.expander('Debug logs', expanded=False):
                    for ln in lines:
                        safe_write(st, ln.rstrip('\n'))
            else:
                safe_write(st, 'Debug logs:')
                for ln in lines:
                    st.sidebar.write(ln.rstrip('\n'))
        except FileNotFoundError:
            safe_write(st, 'Debug logs: (no ipo.debug.log yet)')
    except Exception:
        pass
