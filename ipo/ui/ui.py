import numpy as np
from typing import Any, Optional, List
import numpy as _np

# Re-export selected helpers from ui_sidebar to avoid duplication
from .ui_sidebar import (
    sidebar_metric,
    sidebar_metric_rows,
    compute_step_scores,
    render_iter_step_scores,
    render_mu_value_history,
)


# sidebar_metric and sidebar_metric_rows are imported from ui_sidebar


def render_pair_sidebar(
    lstate,
    prompt: str,
    z_a: np.ndarray,
    z_b: np.ndarray,
    lr_mu_val: float,
    value_scorer=None,
) -> None:
    import streamlit as st  # ensure we use the currently stubbed module in tests
    import numpy as _np

    w_raw = getattr(lstate, "w", None)
    w = (
        _np.asarray(w_raw[: getattr(lstate, "d", 0)], dtype=float).copy()
        if w_raw is not None
        else _np.zeros(getattr(lstate, "d", 0), dtype=float)
    )
    m = pair_metrics(w, z_a, z_b)
    st.sidebar.subheader("Vector info (current pair)")
    sidebar_metric_rows(
        [
            ("‖z_a‖", f"{m['za_norm']:.3f}"),
            ("‖z_b‖", f"{m['zb_norm']:.3f}"),
            ("‖z_b−z_a‖", f"{m['diff_norm']:.3f}"),
        ],
        per_row=2,
    )
    cos = m["cos_w_diff"]
    sidebar_metric_rows(
        [
            (
                "cos(w, z_b−z_a)",
                "n/a"
                if (cos is None or not np.isfinite(float(cos)))
                else f"{float(cos):.3f}",
            )
        ],
        per_row=1,
    )
    from latent_opt import z_from_prompt as _zfp
    z_p = _zfp(lstate, prompt)
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
    sidebar_metric_rows(
        [("V(left)", f"{v_left:.3f}"), ("V(right)", f"{v_right:.3f}")], per_row=2
    )
    mu = lstate.mu
    sidebar_metric_rows(
        [
            ("step(A)", f"{lr_mu_val * float(np.linalg.norm(z_a - mu)):.3f}"),
            ("step(B)", f"{lr_mu_val * float(np.linalg.norm(z_b - mu)):.3f}"),
        ],
        per_row=2,
    )


def env_panel(env: dict) -> None:
    # Keep minimal panel; reuse shared row renderer
    import streamlit as st
    pairs = [("Python", f"{env.get('python')}")]
    cuda = env.get("cuda", "unknown")
    pairs.append(("torch/CUDA", f"{env.get('torch')} | {cuda}"))
    if env.get("streamlit") and env["streamlit"] not in ("unknown", "not imported"):
        pairs.append(("Streamlit", f"{env['streamlit']}") )
    st.sidebar.subheader("Environment")
    sidebar_metric_rows(pairs, per_row=2)


def status_panel(images: tuple, mu_image) -> None:
    from .ui_sidebar import status_panel as _sp
    return _sp(images, mu_image)


def perf_panel(last_call: dict, last_train_ms) -> None:
    # Keep the old entrypoint; leverage ui_sidebar’s row renderer
    import streamlit as st
    pairs = []
    d = last_call.get("dur_s") if isinstance(last_call, dict) else None
    if d is not None:
        pairs.append(("decode_s", f"{float(d):.3f}"))
    if last_train_ms is not None:
        try:
            pairs.append(("train_ms", f"{float(last_train_ms):.1f}"))
        except Exception:
            pass
    if not pairs:
        return
    from .ui_sidebar import sidebar_metric_rows as _smr
    exp = getattr(st.sidebar, "expander", None)
    if callable(exp):
        with exp("Performance", expanded=False):
            _smr(pairs, per_row=2)
    else:
        _smr(pairs, per_row=2)


# compute_step_scores, render_iter_step_scores, and render_mu_value_history
# are imported from ui_sidebar above to keep a single implementation.
