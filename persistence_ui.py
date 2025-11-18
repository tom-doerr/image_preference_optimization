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
    # Export only. Upload has been removed to simplify the UI and avoid
    # cross-prompt/state confusion. Keep the minimal download path.
    st.sidebar.download_button(
        label="Download state (.npz)",
        data=export_state_bytes(lstate, prompt),
        file_name="latent_state.npz",
        mime="application/octet-stream",
    )
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


def render_paths_panel(state_path: str, prompt: str) -> None:
    """Render a minimal 'Paths' panel showing state and dataset NPZ file paths."""
    import os
    import streamlit as st
    from persistence import dataset_path_for_prompt
    try:
        st.sidebar.subheader("Paths")
        st.sidebar.write(f"State path: {state_path} {'(exists)' if os.path.exists(state_path) else '(missing)'}")
        dpath = dataset_path_for_prompt(prompt)
        st.sidebar.write(f"Dataset path: {dpath} {'(exists)' if os.path.exists(dpath) else '(missing)'}")
    except Exception:
        pass


def render_dataset_viewer() -> None:
    """Render a minimal viewer for dataset_*.npz files.

    Shows a selectbox to pick a dataset file, then prints summary
    (rows, dim, pos/neg) and the head of labels.
    """
    import glob
    import numpy as np
    import streamlit as st
    files = sorted(glob.glob('dataset_*.npz'))
    st.sidebar.subheader("Datasets")
    if not files:
        st.sidebar.write("No datasets found")
        return
    sel = None
    if hasattr(st.sidebar, 'selectbox') and callable(getattr(st.sidebar, 'selectbox', None)):
        sel = st.sidebar.selectbox("Dataset file", files, index=0)
    if not sel or sel not in files:
        sel = files[0]
    st.sidebar.write(f"Viewing dataset: {sel}")
    try:
        with np.load(sel) as z:
            X = z['X'] if 'X' in z.files else None
            y = z['y'] if 'y' in z.files else None
        rows = int(X.shape[0]) if X is not None else 0
        d = int(X.shape[1]) if (X is not None and X.ndim == 2) else 0
        pos = int((y > 0).sum()) if y is not None else 0
        neg = int((y < 0).sum()) if y is not None else 0
        st.sidebar.write(f"Rows: {rows}, Dim: {d}, Pos: {pos}, Neg: {neg}")
        if y is not None:
            head = ", ".join([f"{int(v):+d}" for v in np.asarray(y[:10]).astype(int)])
            st.sidebar.write(f"Labels head: [{head}]")
    except Exception as e:
        st.sidebar.write(f"Error reading {sel}: {e}")
