from __future__ import annotations

from typing import Any, Optional
import os


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


def get_log_verbosity(st: Any | None = None) -> int:
    """Return LOG_VERBOSITY as int (0/1/2).

    Order: session_state.log_verbosity → env LOG_VERBOSITY → 0.
    """
    try:
        if st is not None:
            v = getattr(getattr(st, "session_state", st), "log_verbosity", None)
            if v is None:
                v = getattr(st, "log_verbosity", None)
            if v is not None:
                return int(v)
    except Exception:
        pass
    try:
        return int(os.getenv("LOG_VERBOSITY", "0"))
    except Exception:
        return 0


def enable_file_logging(path: str | None = None) -> str:
    """Enable file logging for the shared 'ipo' logger.

    - Path defaults to env IPO_LOG_FILE or 'ipo.debug.log'.
    - Removes existing FileHandlers to avoid duplicates, then adds one.
    - Respects IPO_LOG_LEVEL when present (INFO default).
    Returns the path used.
    """
    import logging
    import os as _os

    log_path = path or _os.getenv("IPO_LOG_FILE") or "ipo.debug.log"
    level_name = (_os.getenv("IPO_LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("ipo")
    logger.setLevel(level)
    # Drop existing file handlers to avoid duplicate writes
    for h in list(logger.handlers):
        try:
            if hasattr(h, "baseFilename"):
                logger.removeHandler(h)
        except Exception:
            pass
    try:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(fh)
    except Exception:
        pass
    return log_path
