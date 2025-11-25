from __future__ import annotations

from typing import Any


def status_panel(st: Any, images: tuple, mu_image) -> None:
    st.sidebar.subheader("Images status")
    left_ready = "ready" if images and images[0] is not None else "empty"
    right_ready = "ready" if images and images[1] is not None else "empty"
    try:
        from ipo.ui.ui import sidebar_metric_rows
        sidebar_metric_rows([("Left", left_ready), ("Right", right_ready)], per_row=2)
    except Exception:
        st.sidebar.write(f"Left: {left_ready}")
        st.sidebar.write(f"Right: {right_ready}")


def env_panel(st: Any, env: dict) -> None:
    st.sidebar.subheader("Environment")
    pairs = [("Python", f"{env.get('python')}")]
    cuda = env.get("cuda", "unknown")
    pairs.append(("torch/CUDA", f"{env.get('torch')} | {cuda}"))
    if env.get("streamlit") and env["streamlit"] not in ("unknown", "not imported"):
        pairs.append(("Streamlit", f"{env['streamlit']}") )
    try:
        from ipo.ui.ui import sidebar_metric_rows
        sidebar_metric_rows(pairs, per_row=2)
    except Exception:
        for k, v in pairs:
            st.sidebar.write(f"{k}: {v}")


def ensure_sidebar_shims(st: Any) -> None:
    """Ensure basic st.sidebar write/metric exist for test stubs."""
    st.sidebar.subheader("Latent state")
    if not hasattr(st.sidebar, "write"):
        def _w(x):
            try:
                if hasattr(st, "sidebar_writes"):
                    st.sidebar_writes.append(str(x))
            except Exception:
                pass
        st.sidebar.write = _w  # type: ignore[attr-defined]
    if not hasattr(st.sidebar, "metric"):
        def _m(label, value, **k):
            try:
                if hasattr(st, "sidebar_writes"):
                    st.sidebar_writes.append(f"{label}: {value}")
            except Exception:
                pass
    st.sidebar.metric = _m  # type: ignore[attr-defined]


def emit_latent_dim_and_data_strip(st: Any, lstate: Any) -> None:
    """Write latent dim and compact pairs/choices strip."""
    try:
        line = f"Latent dim: {int(getattr(lstate, 'd', 0))}"
        if hasattr(st, "sidebar_writes"):
            try:
                st.sidebar_writes.append(line)
            except Exception:
                pass
        st.sidebar.write(line)
    except Exception:
        pass


def emit_step_readouts(st: Any, lstate: Any) -> None:
    """Emit step(A)/step(B) lines derived from current pair and lr_μ.

    Kept minimal for tests; falls back to zeros on failures.
    """
    try:
        import numpy as _np
        mu = getattr(lstate, 'mu', _np.zeros(getattr(lstate, 'd', 0)))
        lr_mu_val = float(getattr(st.session_state, 'lr_mu_ui', getattr(st.session_state, 'LR_MU_UI', 0.3)))
        pair = getattr(st.session_state, 'lz_pair', None)
        if pair is not None:
            z_a, z_b = pair
            sa = lr_mu_val * float(_np.linalg.norm(_np.asarray(z_a) - mu))
            sb = lr_mu_val * float(_np.linalg.norm(_np.asarray(z_b) - mu))
        else:
            sa = sb = 0.0
        st.sidebar.write(f"step(A): {sa:.3f}")
        st.sidebar.write(f"step(B): {sb:.3f}")
    except Exception:
        try:
            st.sidebar.write("step(A): 0.000")
            st.sidebar.write("step(B): 0.000")
        except Exception:
            pass


def emit_debug_panel(st: Any) -> None:
    """Optional Debug panel showing last call and log tail (guarded by checkbox)."""
    try:
        if getattr(st.sidebar, 'checkbox', lambda *a, **k: False)("Debug", value=False):
            from .ui_sidebar_debug import _emit_last_call_info, _emit_log_tail
            _emit_last_call_info(st)
            _emit_log_tail(st)
    except Exception:
        pass


def emit_dim_mismatch(st: Any) -> None:
    """If a dim mismatch is recorded, print an explanatory line."""
    try:
        mismatch = st.session_state.get('dataset_dim_mismatch') or st.session_state.get('DATASET_DIM_MISMATCH')
        if mismatch and isinstance(mismatch, tuple) and len(mismatch) == 2:
            st.sidebar.write(
                f"Dataset recorded at d={mismatch[0]} (ignored); current latent dim d={mismatch[1]}"
            )
    except Exception:
        pass


def emit_last_action_recent(st: Any) -> None:
    try:
        import time as _time
        txt = st.session_state.get('last_action_text') or st.session_state.get('LAST_ACTION_TEXT')
        ts = st.session_state.get('last_action_ts') or st.session_state.get('LAST_ACTION_TS')
        if txt and ts is not None and (_time.time() - float(ts)) < 6.0:
            st.sidebar.write(f"Last action: {txt}")
    except Exception:
        pass


def rows_refresh_tick(st: Any) -> None:
    try:
        rows_live = int(len(st.session_state.get('DATASET_Y', []) or st.session_state.get('dataset_y', []) or []))
    except Exception:
        rows_live = 0
    st.session_state['ROWS_DISPLAY'] = str(rows_live)
    try:
        from ipo.infra.util import get_log_verbosity as _gv
        if int(_gv(st)) >= 1:
            print(f"[rows] live={rows_live} disp={rows_live}")
    except Exception:
        pass
    try:
        from latent_opt import state_summary  # type: ignore
        from ipo.ui.ui import sidebar_metric_rows
        info = state_summary(lstate)
        sidebar_metric_rows([("Pairs:", info.get("pairs_logged", 0)), ("Choices:", info.get("choices_logged", 0))], per_row=2)
    except Exception:
        pass
