from __future__ import annotations

from typing import Callable, Any, Tuple
import numpy as np

__all__ = [
    'get_value_scorer',
    'get_value_scorer_with_status',
]


def get_value_scorer_with_status(vm_choice: str, lstate: Any, prompt: str, session_state: Any) -> Tuple[Callable[[np.ndarray], float], str]:
    """Return a callable f(fvec) -> score based on selected value model.

    - Ridge: dot(w, fvec)
    - XGBoost: uses cached model in session_state.xgb_cache; if unavailable, returns 0
    - DistanceHill: distancehill_score over z = z_p + fvec if dataset exists, else 0
    - CosineHill: cosinehill_score over z = z_p + fvec if dataset exists, else 0
    """
    from latent_logic import z_from_prompt  # local import for tests
    from persistence import get_dataset_for_prompt_or_session

    z_p = z_from_prompt(lstate, prompt)
    # Snapshot w on read to avoid observing a partially-swapped array when
    # Ridge trains asynchronously. Keep it minimal: copy only the used slice.
    _w_raw = getattr(lstate, "w", None)
    w = None if _w_raw is None else np.asarray(_w_raw[: getattr(lstate, 'd', 0)], dtype=float).copy()

    def _ridge(fvec):
        if w is None or w.size == 0:
            return 0.0
        return float(np.dot(w, np.asarray(fvec, dtype=float)))

    def _zero(_fvec):
        return 0.0

    choice = str(vm_choice or "Ridge")
    # Ridge is always available; status is based on whether w is non-zero.
    if choice == "Ridge":
        status = "ridge_untrained"
        try:
            nrm = float(np.linalg.norm(w)) if w is not None else 0.0
            if nrm > 0.0:
                status = "ok"
        except Exception:
            status = "ridge_untrained"
        return _ridge, status

    choice = str(vm_choice or "Ridge")
    if choice == "XGBoost":
        try:
            cache = getattr(session_state, "xgb_cache", {}) or {}
            fut = getattr(session_state, "xgb_fit_future", None)
            if fut is not None and not getattr(fut, "done", lambda: False)() and cache.get("model") is None:
                return _zero, "xgb_training"
            mdl = cache.get("model")
            if mdl is None:
                # Extra context so it's clear why XGB is unavailable.
                try:
                    from persistence import get_dataset_for_prompt_or_session as _get_ds  # type: ignore
                    Xd, yd = _get_ds(prompt, session_state)
                    rows = int(getattr(Xd, "shape", (0, 0))[0]) if Xd is not None else 0
                    d_lat = getattr(lstate, "d", "?")
                    print(f"[xgb] scorer unavailable: no cached model "
                          f"(vm={choice}, dataset_rows={rows}, d={d_lat})")
                except Exception:
                    try:
                        print("[xgb] scorer unavailable: no cached model")
                    except Exception:
                        pass
                return _zero, "xgb_unavailable"
            from xgb_value import score_xgb_proba  # type: ignore
            try:
                n = int(cache.get("n") or 0)
                print(f"[xgb] using cached model rows={n} d={getattr(lstate, 'd', '?')}")
            except Exception:
                pass

            def _xgb(fvec):
                return float(score_xgb_proba(mdl, np.asarray(fvec, dtype=float)))

            return _xgb, "ok"
        except Exception:
            try:
                print("[xgb] scorer error; returning 0 for all scores")
            except Exception:
                pass
            return _zero, "xgb_error"

    if choice == "DistanceHill":
        try:
            X, y = get_dataset_for_prompt_or_session(prompt, session_state)
            if X is None or y is None or getattr(X, "shape", (0,))[0] == 0:
                try:
                    print("[dist] scorer unavailable: empty dataset")
                except Exception:
                    pass
                return _zero, "dist_empty"
            from latent_logic import distancehill_score  # local import

            def _dist(fvec):
                zc = z_p + np.asarray(fvec, dtype=float)
                return float(distancehill_score(prompt, zc, lstate, X, y, gamma=0.5))

            return _dist, "ok"
        except Exception:
            try:
                print("[dist] scorer error; returning 0 for all scores")
            except Exception:
                pass
            return _zero, "dist_error"

    if choice == "CosineHill":
        try:
            X, y = get_dataset_for_prompt_or_session(prompt, session_state)
            if X is None or y is None or getattr(X, "shape", (0,))[0] == 0:
                try:
                    print("[cos] scorer unavailable: empty dataset")
                except Exception:
                    pass
                return _zero, "cos_empty"
            from latent_logic import cosinehill_score  # local import

            def _cos(fvec):
                zc = z_p + np.asarray(fvec, dtype=float)
                return float(cosinehill_score(prompt, zc, lstate, X, y, beta=5.0))

            return _cos, "ok"
        except Exception:
            try:
                print("[cos] scorer error; returning 0 for all scores")
            except Exception:
                pass
            return _zero, "cos_error"

    return _ridge, "ridge_untrained"


def get_value_scorer(vm_choice: str, lstate: Any, prompt: str, session_state: Any) -> Callable[[np.ndarray], float]:
    """Backward-compatible wrapper returning only the scorer."""
    scorer, _status = get_value_scorer_with_status(vm_choice, lstate, prompt, session_state)
    return scorer
