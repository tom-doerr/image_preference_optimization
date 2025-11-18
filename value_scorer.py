from __future__ import annotations

import numpy as np


def get_value_scorer(vm_choice: str, lstate, prompt: str, session_state):
    """Return a callable f(fvec) -> score based on selected value model.

    - Ridge: dot(w, fvec)
    - XGBoost: uses cached model in session_state.xgb_cache, falls back to Ridge
    - DistanceHill: distancehill_score over z = z_p + fvec if dataset exists
    - CosineHill: cosinehill_score over z = z_p + fvec if dataset exists
    """
    from latent_logic import z_from_prompt  # local import for tests
    from persistence import get_dataset_for_prompt_or_session

    z_p = z_from_prompt(lstate, prompt)
    w = getattr(lstate, "w", None)

    def _ridge(fvec):
        if w is None:
            return 0.0
        return float(np.dot(w[: lstate.d], np.asarray(fvec, dtype=float)))

    choice = str(vm_choice or "Ridge")
    if choice == "XGBoost":
        try:
            cache = getattr(session_state, "xgb_cache", {}) or {}
            mdl = cache.get("model")
            if mdl is None:
                return _ridge
            from xgb_value import score_xgb_proba  # type: ignore

            def _xgb(fvec):
                return float(score_xgb_proba(mdl, np.asarray(fvec, dtype=float)))

            return _xgb
        except Exception:
            return _ridge

    if choice == "DistanceHill":
        try:
            X, y = get_dataset_for_prompt_or_session(prompt, session_state)
            if X is None or y is None or getattr(X, "shape", (0,))[0] == 0:
                return _ridge
            from latent_logic import distancehill_score  # local import

            def _dist(fvec):
                zc = z_p + np.asarray(fvec, dtype=float)
                return float(distancehill_score(prompt, zc, lstate, X, y, gamma=0.5))

            return _dist
        except Exception:
            return _ridge

    if choice == "CosineHill":
        try:
            X, y = get_dataset_for_prompt_or_session(prompt, session_state)
            if X is None or y is None or getattr(X, "shape", (0,))[0] == 0:
                return _ridge
            from latent_logic import cosinehill_score  # local import

            def _cos(fvec):
                zc = z_p + np.asarray(fvec, dtype=float)
                return float(cosinehill_score(prompt, zc, lstate, X, y, beta=5.0))

            return _cos
        except Exception:
            return _ridge

    return _ridge

