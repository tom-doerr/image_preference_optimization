from __future__ import annotations

from typing import Any


def render_iter_step_scores(
    st: Any,
    lstate: Any,
    prompt: str,
    vm_choice: str,
    iter_steps: int,
    iter_eta: float | None,
    trust_r: float | None,
) -> None:
    try:
        from .ui_sidebar import sidebar_metric_rows
        from .ui_step_scores import compute_step_scores as _css
        scores = _css(lstate, prompt, vm_choice, iter_steps, iter_eta, trust_r, st.session_state)
        if scores is None:
            st.sidebar.write("Step scores: n/a")
            sidebar_metric_rows([("Step scores", "n/a")], per_row=1)
            return
        st.sidebar.write("Step scores: " + ", ".join(f"{v:.3f}" for v in scores[:8]))
        pairs = [(f"Step {i}", f"{v:.3f}") for i, v in enumerate(scores[:4], 1)]
        sidebar_metric_rows(pairs, per_row=2)
    except Exception:
        try:
            st.sidebar.write("Step scores: n/a")
        except Exception:
            pass
