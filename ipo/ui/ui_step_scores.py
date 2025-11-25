from __future__ import annotations

from typing import Any


def compute_step_scores(
    lstate: Any,
    prompt: str,
    vm_choice: str,
    iter_steps: int,
    iter_eta: float | None,
    trust_r: float | None,
    session_state: Any,
):
    """Compute step scores along the ridge direction using the active scorer.

    Delegates the step-length heuristic to ui_sidebar._step_len_for_scores to keep
    a single source of truth for eta/trust radius behavior.
    Returns a list of floats or None when prerequisites are missing.
    """
    try:
        from ipo.ui.ui_sidebar import _step_len_for_scores  # reuse existing helper
    except Exception:  # pragma: no cover - defensive import for test stubs
        def _step_len_for_scores(_ls, _n, _e, _r):  # type: ignore
            return 1.0 / max(1, int(_n))

    try:
        import numpy as _np
        from latent_logic import z_from_prompt as _zfp
        from value_scorer import get_value_scorer as _gvs

        # Ridge direction (unit)
        w_raw = getattr(lstate, "w", None)
        if w_raw is None:
            return None
        d = int(getattr(lstate, "d", 0))
        w = _np.asarray(w_raw[:d], dtype=float)
        nrm = float(_np.linalg.norm(w))
        if nrm == 0.0:
            return None
        d1 = w / nrm

        # Scorer selection (None → unavailable unless Ridge)
        scorer, tag = _gvs(vm_choice, lstate, prompt, session_state)
        if vm_choice != "Ridge" and scorer is None:
            return None

        # Step sizes and anchor
        n_steps = max(1, int(iter_steps))
        step_len = float(_step_len_for_scores(lstate, n_steps, iter_eta, trust_r))
        z_p = _zfp(lstate, prompt)

        scores: list[float] = []
        for k in range(1, n_steps + 1):
            zc = z_p + (k * step_len) * d1
            fvec = zc - z_p
            try:
                s = float(scorer(fvec)) if scorer is not None else float(_np.dot(w, fvec))
            except Exception:
                s = float(_np.dot(w, fvec))
            scores.append(s)
        return scores
    except Exception:
        return None

