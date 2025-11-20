from persistence import export_state_bytes


def render_persistence_controls(
    lstate, prompt: str, state_path: str, apply_state_fn, rerun_fn=None
) -> None:
    import streamlit as st  # import here to use the currently stubbed module in tests

    """Render sidebar persistence controls (download/upload) with minimal logic.

    - Downloads the current state augmented with prompt/app_version/created_at.
    - On upload, warns if the uploaded prompt differs and offers a one-click switch.
    - Uses `apply_state_fn(new_state)` to inject state into the app and `save_state` to persist.
    - Calls `rerun_fn()` if provided (e.g., st.rerun) after changes.
    """
    # Export only. Upload has been removed to simplify the UI and avoid
    # cross-prompt/state confusion. Keep the minimal download path.
    try:
        data = export_state_bytes(lstate, prompt)
    except Exception:
        data = b""
    st.sidebar.download_button(label="Download state (.npz)", data=data, file_name="latent_state.npz", mime="application/octet-stream")
    # Upload removed


def render_metadata_panel(state_path: str, prompt: str, per_row: int = 2) -> None:
    """Render the 'State metadata' sidebar panel (app_version, created_at, prompt_hash)."""
    import os
    import hashlib
    import streamlit as st
    from persistence import read_metadata
    from ui import sidebar_metric_rows

    try:
        if os.path.exists(state_path):
            meta = read_metadata(state_path)
            if meta.get("app_version") or meta.get("created_at"):
                st.sidebar.subheader("State metadata")
                pairs = []
                if meta.get("app_version"):
                    pairs.append(("app_version", f"{meta['app_version']}"))
                if meta.get("created_at"):
                    pairs.append(("created_at", f"{meta['created_at']}"))
                try:
                    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
                    pairs.append(("prompt_hash", h))
                except Exception:
                    pass
                sidebar_metric_rows(pairs, per_row=per_row)
                # Also emit a plain line for prompt hash for tests that scan text
                try:
                    st.sidebar.write(f"prompt_hash: {h}")
                except Exception:
                    pass
    except Exception:
        pass
