import numpy as np
from metrics import pair_metrics
from latent_opt import z_from_prompt


def sidebar_metric(label: str, value) -> None:
    import streamlit as st  # ensure we use the currently stubbed module in tests
    try:
        if hasattr(st.sidebar, "metric") and callable(getattr(st.sidebar, "metric", None)):
            st.sidebar.metric(label, str(value))
        else:
            st.sidebar.write(f"{label}: {value}")
    except Exception:
        st.sidebar.write(f"{label}: {value}")


def sidebar_metric_rows(pairs, per_row: int = 2) -> None:
    import streamlit as st  # ensure we use the currently stubbed module in tests
    try:
        for i in range(0, len(pairs), per_row):
            row = pairs[i : i + per_row]
            if hasattr(st.sidebar, "columns") and callable(getattr(st.sidebar, "columns", None)) and len(row) > 1:
                cols = st.sidebar.columns(len(row))
                for (label, value), col in zip(row, cols):
                    with col:
                        sidebar_metric(label, value)
            else:
                for label, value in row:
                    sidebar_metric(label, value)
    except Exception:
        for label, value in pairs:
            sidebar_metric(label, value)


def render_pair_sidebar(lstate, prompt: str, z_a: np.ndarray, z_b: np.ndarray, lr_mu_val: float, value_scorer=None) -> None:
    import streamlit as st  # ensure we use the currently stubbed module in tests
    import numpy as _np
    w_raw = getattr(lstate, 'w', None)
    w = _np.asarray(w_raw[: getattr(lstate, 'd', 0)], dtype=float).copy() if w_raw is not None else _np.zeros(getattr(lstate, 'd', 0), dtype=float)
    m = pair_metrics(w, z_a, z_b)
    st.sidebar.subheader("Vector info (current pair)")
    sidebar_metric_rows(
        [("‖z_a‖", f"{m['za_norm']:.3f}"), ("‖z_b‖", f"{m['zb_norm']:.3f}"), ("‖z_b−z_a‖", f"{m['diff_norm']:.3f}")],
        per_row=2,
    )
    cos = m["cos_w_diff"]
    sidebar_metric_rows(
        [("cos(w, z_b−z_a)", "n/a" if (cos is None or not np.isfinite(float(cos))) else f"{float(cos):.3f}")],
        per_row=1,
    )
    z_p = z_from_prompt(lstate, prompt)
    sidebar_metric_rows(
        [
            ("‖μ−z_prompt‖", f"{float(np.linalg.norm(lstate.mu - z_p)):.3f}"),
            ("‖z_a−z_prompt‖", f"{float(np.linalg.norm(z_a - z_p)):.3f}"),
            ("‖z_b−z_prompt‖", f"{float(np.linalg.norm(z_b - z_p)):.3f}"),
        ],
        per_row=2,
    )
    if value_scorer is not None:
        v_left = float(value_scorer(z_a - z_p))
        v_right = float(value_scorer(z_b - z_p))
    else:
        v_left = float(np.dot(w, (z_a - z_p)))
        v_right = float(np.dot(w, (z_b - z_p)))
    sidebar_metric_rows([("V(left)", f"{v_left:.3f}"), ("V(right)", f"{v_right:.3f}")], per_row=2)
    mu = lstate.mu
    sidebar_metric_rows(
        [("step(A)", f"{lr_mu_val * float(np.linalg.norm(z_a - mu)):.3f}"), ("step(B)", f"{lr_mu_val * float(np.linalg.norm(z_b - mu)):.3f}")],
        per_row=2,
    )


def env_panel(env: dict) -> None:
    import streamlit as st
    pairs = [("Python", f"{env.get('python')}")]
    cuda = env.get('cuda', 'unknown')
    pairs.append(("torch/CUDA", f"{env.get('torch')} | {cuda}"))
    if env.get('streamlit') and env['streamlit'] not in ('unknown', 'not imported'):
        pairs.append(("Streamlit", f"{env['streamlit']}"))
    st.sidebar.subheader("Environment")
    sidebar_metric_rows(pairs, per_row=2)


def status_panel(images: tuple, mu_image) -> None:
    import streamlit as st
    st.sidebar.subheader("Images status")
    left_ready = 'ready' if images and images[0] is not None else 'empty'
    right_ready = 'ready' if images and images[1] is not None else 'empty'
    sidebar_metric_rows([("Left", left_ready), ("Right", right_ready)], per_row=2)


def perf_panel(last_call: dict, last_train_ms) -> None:
    """Render a minimal Performance panel (decode_s, train_ms)."""
    import streamlit as st
    pairs = []
    d = last_call.get('dur_s') if isinstance(last_call, dict) else None
    if d is not None:
        pairs.append(("decode_s", f"{float(d):.3f}"))
    if last_train_ms is not None:
        try:
            pairs.append(("train_ms", f"{float(last_train_ms):.1f}"))
        except Exception:
            pass
    if not pairs:
        return
    exp = getattr(st.sidebar, 'expander', None)
    if callable(exp):
        with exp("Performance", expanded=False):
            sidebar_metric_rows(pairs, per_row=2)
    else:
        sidebar_metric_rows(pairs, per_row=2)
