from __future__ import annotations

from typing import Callable, Any, Tuple
import numpy as np

__all__ = [
    "get_value_scorer",
    "get_value_scorer_with_status",
]


def _snapshot_w(lstate: Any) -> np.ndarray | None:
    """Return a copy of w[:d] or None when unavailable."""
    try:
        _w_raw = getattr(lstate, "w", None)
        if _w_raw is None:
            return None
        d = int(getattr(lstate, "d", len(_w_raw)))
        return np.asarray(_w_raw[:d], dtype=float).copy()
    except Exception:
        return None


def _build_ridge_scorer(lstate: Any) -> Tuple[Callable[[np.ndarray], float], str]:
    w = _snapshot_w(lstate)

    def _ridge(fvec: np.ndarray) -> float:
        if w is None or w.size == 0:
            return 0.0
        return float(np.dot(w, np.asarray(fvec, dtype=float)))

    try:
        nrm = float(np.linalg.norm(w)) if w is not None else 0.0
        return _ridge, ("ok" if nrm > 0.0 else "ridge_untrained")
    except Exception:
        return _ridge, "ridge_untrained"


def _build_xgb_scorer(
    vm_choice: str, lstate: Any, prompt: str, session_state: Any
) -> Tuple[Callable[[np.ndarray], float], str]:
    def _zero(_fvec: np.ndarray) -> float:
        return 0.0

    try:
        cache = getattr(session_state, "xgb_cache", {}) or {}
        fut = getattr(session_state, "xgb_fit_future", None)
        if (
            fut is not None
            and not getattr(fut, "done", lambda: False)()
            and cache.get("model") is None
        ):
            return _zero, "xgb_training"
        mdl = cache.get("model")
        if mdl is None:
            try:
                from xgb_value import get_cached_scorer  # type: ignore

                scorer = get_cached_scorer(prompt, session_state)
                if scorer is not None:
                    return scorer, "ok"
            except Exception:
                pass
        if mdl is None:
            try:
                from persistence import get_dataset_for_prompt_or_session as _get_ds  # type: ignore

                Xd, _ = _get_ds(prompt, session_state)
                rows = int(getattr(Xd, "shape", (0, 0))[0]) if Xd is not None else 0
                d_lat = getattr(lstate, "d", "?")
                print(
                    f"[xgb] scorer unavailable: no cached model (vm={vm_choice}, dataset_rows={rows}, d={d_lat})"
                )
            except Exception:
                print("[xgb] scorer unavailable: no cached model")
            return _zero, "xgb_unavailable"
        from xgb_value import score_xgb_proba  # type: ignore

        try:
            n = int(cache.get("n") or 0)
            print(f"[xgb] using cached model rows={n} d={getattr(lstate, 'd', '?')}")
        except Exception:
            pass

        def _xgb(fvec: np.ndarray) -> float:
            return float(score_xgb_proba(mdl, np.asarray(fvec, dtype=float)))

        return _xgb, "ok"
    except Exception:
        print("[xgb] scorer error; returning 0 for all scores")
        return _zero, "xgb_error"


def _build_dist_scorer(
    kind: str, lstate: Any, prompt: str, session_state: Any
) -> Tuple[Callable[[np.ndarray], float], str]:
    def _zero(_fvec: np.ndarray) -> float:
        return 0.0

    try:
        from persistence import get_dataset_for_prompt_or_session

        X, y = get_dataset_for_prompt_or_session(prompt, session_state)
        if X is None or y is None or getattr(X, "shape", (0,))[0] == 0:
            print(f"[{kind}] scorer unavailable: empty dataset")
            return _zero, f"{'dist' if kind == 'dist' else 'cos'}_empty"
        from latent_logic import z_from_prompt

        z_p = z_from_prompt(lstate, prompt)
        if kind == "dist":
            from latent_logic import distancehill_score

            def _dist(fvec: np.ndarray) -> float:
                zc = z_p + np.asarray(fvec, dtype=float)
                return float(distancehill_score(prompt, zc, lstate, X, y, gamma=0.5))

            return _dist, "ok"
        else:
            from latent_logic import cosinehill_score

            def _cos(fvec: np.ndarray) -> float:
                zc = z_p + np.asarray(fvec, dtype=float)
                return float(cosinehill_score(prompt, zc, lstate, X, y, beta=5.0))

            return _cos, "ok"
    except Exception:
        print(f"[{kind}] scorer error; returning 0 for all scores")
        return _zero, f"{'dist' if kind == 'dist' else 'cos'}_error"


def get_value_scorer_with_status(
    vm_choice: str, lstate: Any, prompt: str, session_state: Any
) -> Tuple[Callable[[np.ndarray], float], str]:
    """Return a callable f(fvec) -> score and a status string.

    Split into small builders per value model to keep this dispatcher simple.
    """
    choice = str(vm_choice or "Ridge")
    if choice == "Ridge":
        return _build_ridge_scorer(lstate)
    if choice == "XGBoost":
        return _build_xgb_scorer(choice, lstate, prompt, session_state)
    if choice == "DistanceHill":
        return _build_dist_scorer("dist", lstate, prompt, session_state)
    if choice == "CosineHill":
        return _build_dist_scorer("cos", lstate, prompt, session_state)
    # Fallback: Ridge semantics
    return _build_ridge_scorer(lstate)


def get_value_scorer(
    vm_choice: str, lstate: Any, prompt: str, session_state: Any
) -> Callable[[np.ndarray], float]:
    """Backward-compatible wrapper returning only the scorer."""
    scorer, _status = get_value_scorer_with_status(
        vm_choice, lstate, prompt, session_state
    )
    return scorer
