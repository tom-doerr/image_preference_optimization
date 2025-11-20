from __future__ import annotations

from typing import Any
import numpy as np


def render_iter_step_scores(
    st: Any,
    lstate: Any,
    prompt: str,
    vm_choice: str,
    iter_steps: int,
    iter_eta: float | None,
    trust_r: float | None,
) -> None:
    """Render per-step latent optimization scores into the sidebar.

    Minimal dependency surface; reads dataset via persistence helper and uses
    latent_logic scorers or Ridge/XGB depending on `vm_choice`.
    """
    try:
        from latent_logic import z_from_prompt  # type: ignore
        # Gather weights and scorer status
        w = getattr(lstate, 'w', None)
        w = w[: lstate.d] if w is not None else None
        n = float(np.linalg.norm(w)) if w is not None else 0.0
        scorer = None
        status = "ok"
        try:
            from value_scorer import get_value_scorer_with_status as _gvs
            scorer, status = _gvs(vm_choice, lstate, prompt, st.session_state)
        except Exception:
            try:
                from value_scorer import get_value_scorer as _gvs2
                scorer = _gvs2(vm_choice, lstate, prompt, st.session_state)
                status = "ok"
            except Exception:
                scorer = None
                status = "unavailable"

        # If no usable weights/scorer, prefer showing n/a rather than zeros
        if (vm_choice == "Ridge" and (w is None or n == 0.0)) or (vm_choice != "Ridge" and status != "ok"):
            st.sidebar.write("Step scores: n/a")
            try:
                from ui import sidebar_metric_rows
                sidebar_metric_rows([("Step scores", "n/a")], per_row=1)
            except Exception:
                pass
            return

        # Compute step scores along d1 ∥ w
        d1 = w / n
        n_steps = max(1, int(iter_steps))
        if iter_eta is not None and float(iter_eta) > 0.0:
            step_len = float(iter_eta)
        elif trust_r and float(trust_r) > 0.0:
            step_len = float(trust_r) / n_steps
        else:
            step_len = float(lstate.sigma) / n_steps
        z_p = z_from_prompt(lstate, prompt)
        scores: list[float] = []
        for k in range(1, n_steps + 1):
            zc = z_p + (k * step_len) * d1
            try:
                if scorer is not None:
                    s = float(scorer(zc - z_p))
                else:
                    s = float(np.dot(lstate.w[: lstate.d], (zc - z_p)))
            except Exception:
                s = float(np.dot(lstate.w[: lstate.d], (zc - z_p)))
            scores.append(s)

        try:
            st.sidebar.write("Step scores: " + ", ".join(f"{v:.3f}" for v in scores[:8]))
        except Exception:
            pass
        try:
            from ui import sidebar_metric_rows
            pairs = [(f"Step {i}", f"{v:.3f}") for i, v in enumerate(scores[:4], 1)]
            sidebar_metric_rows(pairs, per_row=2)
        except Exception:
            pass
    except Exception:
        pass


def render_mu_value_history(st: Any, lstate: Any, prompt: str) -> None:
    """Plot ‖μ−z_prompt‖ per optimization step when history is available."""
    try:
        mu_hist = getattr(lstate, "mu_hist", None)
        if mu_hist is None or getattr(mu_hist, "size", 0) == 0:
            return
        from latent_logic import z_from_prompt
        z_p = z_from_prompt(lstate, prompt).reshape(1, -1)
        mu_flat = mu_hist.reshape(mu_hist.shape[0], -1)
        vals = np.linalg.norm(mu_flat - z_p, axis=1)
        sb = getattr(st, "sidebar", st)
        if hasattr(sb, "line_chart"):
            sb.subheader("Latent distance per step")
            sb.line_chart(vals.tolist())
    except Exception:
        pass
