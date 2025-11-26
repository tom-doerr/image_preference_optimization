from __future__ import annotations

from typing import Any


def resolve_meta_pairs(prompt: str, state_path: str):
    try:
        import hashlib
        import os

        from ipo.core.persistence import read_metadata
        meta = None
        path = state_path
        if os.path.exists(path):
            meta = read_metadata(path)
        else:
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
            alt = os.path.join("data", h, "latent_state.npz")
            if os.path.exists(alt):
                path = alt
                meta = read_metadata(path)
        if not meta or not (meta.get("app_version") or meta.get("created_at")):
            return None
        pairs = []
        if meta.get("app_version"):
            pairs.append(("app_version", f"{meta['app_version']}"))
        if meta.get("created_at"):
            pairs.append(("created_at", f"{meta['created_at']}"))
        ph = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        pairs.append(("prompt_hash", ph))
        return pairs
    except Exception:
        return None


def emit_meta_pairs(st: Any, pairs) -> None:
    try:
        st.sidebar.subheader("State metadata")
    except Exception:
        pass
    try:
        from ipo.ui.ui import sidebar_metric_rows
        sidebar_metric_rows(pairs, per_row=2)
    except Exception:
        try:
            for k, v in (pairs or []):
                st.sidebar.write(f"{k}: {v}")
        except Exception:
            pass
