from __future__ import annotations

from typing import Any, Optional


def safe_set(ns: Any, key: str, value: Any) -> None:
    """Set a key in a dict-like session_state; ignore failures.

    Minimal helper to trim repeated try/except blocks.
    """
    try:
        ns[key] = value
    except Exception:
        try:
            setattr(ns, key, value)  # fallback for attr-style stubs
        except Exception:
            pass


def safe_sidebar_num(
    st: Any,
    label: str,
    *,
    value: float | int,
    step: Optional[float | int] = None,
    format: Optional[str] = None,
):
    """Call sidebar.number_input if available; otherwise return the given value.

    Keeps behavior minimal and deterministic for stubs.
    """
    num = getattr(getattr(st, "sidebar", st), "number_input", getattr(st, "number_input", None))
    if callable(num):
        try:
            return num(label, value=value, step=step, format=format)
        except Exception:
            return value
    return value


def safe_write(st: Any, line: Any) -> None:
    """Write a line to st.sidebar and capture into st.sidebar_writes when present.

    Keeps sidebar text assertions stable under simple Streamlit stubs.
    """
    try:
        if hasattr(st, "sidebar_writes"):
            st.sidebar_writes.append(str(line))
    except Exception:
        pass
    try:
        sb = getattr(st, "sidebar", None)
        w = getattr(sb, "write", None)
        if callable(w):
            w(str(line))
    except Exception:
        pass
