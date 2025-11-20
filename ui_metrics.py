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
        # Minimal local import to avoid missing-name crash; keeps deps light.
        from latent_logic import z_from_prompt  # type: ignore
        try:
            from persistence import get_dataset_for_prompt_or_session
            get_dataset_for_prompt_or_session(prompt, st.session_state)
        except Exception:
            pass
        w = getattr(lstate, 'w', None)
        w = w[: lstate.d] if w is not None else None
        n = float(np.linalg.norm(w)) if w is not None else 0.0
        d1 = w / n
        n_steps = max(1, int(iter_steps))
        if iter_eta is not None and float(iter_eta) > 0.0:
            step_len = float(iter_eta)
        elif trust_r and float(trust_r) > 0.0:
            step_len = float(trust_r) / n_steps
        else:
            step_len = float(lstate.sigma) / n_steps
        z_p = z_from_prompt(lstate, prompt)
        try:
            from value_scorer import get_value_scorer as _get_vs
            scorer = _get_vs(vm_choice, lstate, prompt, st.session_state)
        except Exception:
            scorer = None
        scores: list[float] = []
        for k in range(1, n_steps + 1):
            if w is None or n == 0.0:
                scores.append(0.0)
                continue
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
            if scores:
                st.sidebar.write("Step scores: " + ", ".join(f"{v:.3f}" for v in scores[:8]))
            else:
                st.sidebar.write("Step scores: n/a")
        except Exception:
            pass
        try:
            from ui import sidebar_metric_rows
            pairs = [(f"Step {i}", f"{v:.3f}") for i, v in enumerate(scores[:4], 1)] if scores else [("Step scores", "n/a")]
            if pairs:
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
