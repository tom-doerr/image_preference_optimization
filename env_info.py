import sys


def get_env_summary():
    info = {"python": sys.version.split()[0]}
    # torch (optional)
    try:
        import torch  # type: ignore
        info["torch"] = getattr(torch, "__version__", "unknown")
        try:
            info["cuda"] = str(bool(getattr(torch.cuda, "is_available", lambda: False)()))
        except Exception:
            info["cuda"] = "unknown"
    except Exception:
        info["torch"] = "not installed"
        info["cuda"] = "unknown"
    # streamlit (optional)
    try:
        import streamlit as st  # type: ignore
        info["streamlit"] = getattr(st, "__version__", "unknown")
    except Exception:
        info["streamlit"] = "not imported"
    return info

