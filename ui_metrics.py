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
        from persistence import get_dataset_for_prompt_or_session
        Xd, yd = get_dataset_for_prompt_or_session(prompt, st.session_state)
        if Xd is None or yd is None or len(getattr(yd, 'shape', (0,))) == 0:
            return
        w = getattr(lstate, 'w', None)
        if w is None:
            return
        w = w[: lstate.d]
        n = float(np.linalg.norm(w))
        if n == 0.0:
            return
        d1 = w / n
        n_steps = max(1, int(iter_steps))
        if iter_eta is not None and float(iter_eta) > 0.0:
            step_len = float(iter_eta)
        elif trust_r and float(trust_r) > 0.0:
            step_len = float(trust_r) / n_steps
        else:
            step_len = float(lstate.sigma) / n_steps
        from latent_logic import z_from_prompt, distancehill_score, cosinehill_score
        from constants import DISTANCEHILL_GAMMA, COSINEHILL_BETA
        z_p = z_from_prompt(lstate, prompt)
        scores: list[float] = []
        for k in range(1, n_steps + 1):
            zc = z_p + (k * step_len) * d1
            if vm_choice == 'CosineHill':
                s = float(cosinehill_score(prompt, zc, lstate, Xd, yd, beta=COSINEHILL_BETA))
            elif vm_choice == 'DistanceHill':
                s = float(distancehill_score(prompt, zc, lstate, Xd, yd, gamma=DISTANCEHILL_GAMMA))
            elif vm_choice == 'XGBoost':
                try:
                    from xgb_value import score_xgb_proba
                    s = float(score_xgb_proba((st.session_state.get('xgb_cache') or {}).get('model'), (zc - z_p)))
                except Exception:
                    s = 0.0
            else:
                s = float(np.dot(lstate.w[: lstate.d], (zc - z_p)))
            scores.append(s)
        try:
            st.sidebar.write("Step scores: " + ", ".join(f"{v:.3f}" for v in scores[:8]))
        except Exception:
            pass
        try:
            from ui import sidebar_metric_rows
            pairs = [(f"Step {i}", f"{v:.3f}") for i, v in enumerate(scores[:4], 1)]
            if pairs:
                sidebar_metric_rows(pairs, per_row=2)
        except Exception:
            pass
    except Exception:
        pass

