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
    try:
        from latent_opt import state_summary  # type: ignore
        from ipo.ui.ui import sidebar_metric_rows
        info = state_summary(lstate)
        sidebar_metric_rows([("Pairs:", info.get("pairs_logged", 0)), ("Choices:", info.get("choices_logged", 0))], per_row=2)
    except Exception:
        pass
