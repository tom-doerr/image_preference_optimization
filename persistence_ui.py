import io
import numpy as np

from persistence import state_path_for_prompt, export_state_bytes
from latent_opt import loads_state, save_state


def render_persistence_controls(lstate, prompt: str, state_path: str, apply_state_fn, rerun_fn=None) -> None:
    import streamlit as st  # import here to use the currently stubbed module in tests
    """Render sidebar persistence controls (download/upload) with minimal logic.

    - Downloads the current state augmented with prompt/app_version/created_at.
    - On upload, warns if the uploaded prompt differs and offers a one-click switch.
    - Uses `apply_state_fn(new_state)` to inject state into the app and `save_state` to persist.
    - Calls `rerun_fn()` if provided (e.g., st.rerun) after changes.
    """
    st.sidebar.download_button(
        label="Download state (.npz)",
        data=export_state_bytes(lstate, prompt),
        file_name="latent_state.npz",
        mime="application/octet-stream",
    )
    uploaded = st.sidebar.file_uploader("Upload state (.npz)", type=["npz"])
    if uploaded is not None and st.sidebar.button("Load uploaded state"):
        data_bytes = uploaded.read()
        up_prompt = None
        try:
            arr = np.load(io.BytesIO(data_bytes))
            if 'prompt' in arr.files:
                up_prompt = arr['prompt'].item()
        except Exception:
            up_prompt = None
        if up_prompt is not None and up_prompt != prompt:
            st.sidebar.warning(
                f"Uploaded state is for a different prompt: '{up_prompt}'. Change the Prompt or switch via Manage states, then load."
            )
            if st.sidebar.button("Switch to uploaded prompt and load now"):
                # Switch prompt and load uploaded state immediately
                st.session_state.prompt = up_prompt
                st.session_state.state_path = state_path_for_prompt(up_prompt)
                new_state = loads_state(data_bytes)
                apply_state_fn(new_state)
                save_state(new_state, st.session_state.state_path)
            if callable(rerun_fn):
                rerun_fn()
        else:
            new_state = loads_state(data_bytes)
            apply_state_fn(new_state)
            save_state(new_state, state_path)
            if callable(rerun_fn):
                rerun_fn()


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
            if meta.get('app_version') or meta.get('created_at'):
                st.sidebar.subheader("State metadata")
                pairs = []
                if meta.get('app_version'):
                    pairs.append(("app_version", f"{meta['app_version']}"))
                if meta.get('created_at'):
                    pairs.append(("created_at", f"{meta['created_at']}"))
                try:
                    h = hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:10]
                    pairs.append(("prompt_hash", h))
                except Exception:
                    pass
                sidebar_metric_rows(pairs, per_row=per_row)
    except Exception:
        pass
