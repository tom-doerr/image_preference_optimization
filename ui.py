import numpy as np
from typing import Any, Optional, List
import numpy as _np


def sidebar_metric(label: str, value) -> None:
    # Delegate to ui_sidebar to keep a single implementation
    from ui_sidebar import sidebar_metric as _sm

    return _sm(label, value)


def sidebar_metric_rows(pairs, per_row: int = 2) -> None:
    from ui_sidebar import sidebar_metric_rows as _smr

    return _smr(pairs, per_row=per_row)


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
    from ui_sidebar import sidebar_metric_rows as _smr
    _smr(pairs, per_row=2)


def status_panel(images: tuple, mu_image) -> None:
    from ui_sidebar import status_panel as _sp

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
    from ui_sidebar import sidebar_metric_rows as _smr
    exp = getattr(st.sidebar, "expander", None)
    if callable(exp):
        with exp("Performance", expanded=False):
            _smr(pairs, per_row=2)
    else:
        _smr(pairs, per_row=2)


# Collapsed from ui_metrics.py (195d)
def compute_step_scores(
    lstate: Any,
    prompt: str,
    vm_choice: str,
    iter_steps: int,
    iter_eta: Optional[float],
    trust_r: Optional[float],
    session_state: Any,
) -> Optional[List[float]]:
    try:
        from latent_logic import z_from_prompt as _zfp  # lazy import
        w_raw = getattr(lstate, "w", None)
        w = (
            None
            if w_raw is None
            else _np.asarray(w_raw[: getattr(lstate, "d", 0)], dtype=float).copy()
        )
        n = float(_np.linalg.norm(w)) if w is not None else 0.0
        scorer = None
        status = "ok"
        try:
            from value_scorer import get_value_scorer as _gvs
            scorer, tag_or_status = _gvs(vm_choice, lstate, prompt, session_state)
            status = "ok" if scorer is not None else str(tag_or_status)
        except Exception:
            scorer = None
            status = "unavailable"
        if w is None or n == 0.0:
            return None
        if vm_choice != "Ridge" and status != "ok":
            return None
        d1 = w / n
        n_steps = max(1, int(iter_steps))
        if iter_eta is not None and float(iter_eta) > 0.0:
            step_len = float(iter_eta)
        elif trust_r and float(trust_r) > 0.0:
            step_len = float(trust_r) / n_steps
        else:
            step_len = float(getattr(lstate, "sigma", 1.0)) / n_steps
        z_p = _zfp(lstate, prompt)
        scores: list[float] = []
        for k in range(1, n_steps + 1):
            zc = z_p + (k * step_len) * d1
            try:
                if scorer is not None:
                    s = float(scorer(zc - z_p))
                else:
                    s = float(_np.dot(w, (zc - z_p)))
            except Exception:
                s = float(_np.dot(w, (zc - z_p)))
            scores.append(s)
        return scores
    except Exception:
        return None


def render_iter_step_scores(
    st: Any,
    lstate: Any,
    prompt: str,
    vm_choice: str,
    iter_steps: int,
    iter_eta: float | None,
    trust_r: float | None,
) -> None:
    scores = compute_step_scores(
        lstate, prompt, vm_choice, iter_steps, iter_eta, trust_r, st.session_state
    )
    if scores is None:
        try:
            st.sidebar.write("Step scores: n/a")
            sidebar_metric_rows([("Step scores", "n/a")], per_row=1)
        except Exception:
            pass
        return
    try:
        st.sidebar.write("Step scores: " + ", ".join(f"{v:.3f}" for v in scores[:8]))
    except Exception:
        pass
    try:
        pairs = [(f"Step {i}", f"{v:.3f}") for i, v in enumerate(scores[:4], 1)]
        sidebar_metric_rows(pairs, per_row=2)
    except Exception:
        pass


def render_mu_value_history(st: Any, lstate: Any, prompt: str) -> None:
    try:
        mu_hist = getattr(lstate, "mu_hist", None)
        if mu_hist is None or getattr(mu_hist, "size", 0) == 0:
            return
        from latent_opt import z_from_prompt as _zfp
        z_p = _zfp(lstate, prompt).reshape(1, -1)
        mu_flat = mu_hist.reshape(mu_hist.shape[0], -1)
        vals = _np.linalg.norm(mu_flat - z_p, axis=1)
        sb = getattr(st, "sidebar", st)
        if hasattr(sb, "line_chart"):
            sb.subheader("Latent distance per step")
            sb.line_chart(vals.tolist())
    except Exception:
        pass
