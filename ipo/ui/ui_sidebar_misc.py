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

