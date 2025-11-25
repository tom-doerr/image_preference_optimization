from __future__ import annotations

from typing import Callable, Any, Tuple
import numpy as np

__all__ = [
    "get_value_scorer",  # unified API: returns (scorer|None, tag_or_status)
    "get_value_scorer_with_status",  # backwards-compat shim
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
        try:
            print(f"[ridge-scorer] ||w||={nrm:.3f} status={'ok' if nrm>0 else 'ridge_untrained'}")
        except Exception:
            pass
        return _ridge, ("ok" if nrm > 0.0 else "ridge_untrained")
    except Exception:
        return _ridge, "ridge_untrained"


def _build_distance_scorer(session_state: Any) -> Tuple[Callable[[np.ndarray], float], str]:
    try:
        from ipo.infra.constants import Keys as _K

        p = float(getattr(session_state, _K.DIST_EXP, session_state.get(_K.DIST_EXP, 2.0)))
    except Exception:
        p = 2.0

    def _dist(fvec: np.ndarray) -> float:
        fv = np.asarray(fvec, dtype=float)
        try:
            if p == 2.0:
                return float(-np.sum(fv * fv))
            if p == 1.0:
                return float(-np.sum(np.abs(fv)))
            return float(-np.sum(np.abs(fv) ** p))
        except Exception:
            return float(-np.sum(fv * fv))

    try:
        print(f"[dist] exponent p={p}")
    except Exception:
        pass
    return _dist, "Distance"


def _build_logit_scorer(session_state: Any) -> Tuple[Callable[[np.ndarray], float] | None, str]:
    try:
        from ipo.infra.constants import Keys as _K
        w = session_state.get(_K.LOGIT_W)
        if w is None:
            return None, "logit_untrained"
        w = np.asarray(w, dtype=float)
        if w.size == 0 or float(np.linalg.norm(w)) == 0.0:
            return None, "logit_untrained"

        def _logit(fvec: np.ndarray) -> float:
            z = float(np.dot(w, np.asarray(fvec, dtype=float)))
            return float(1.0 / (1.0 + np.exp(-z)))

        try:
            print(f"[logit] scorer ready ||w||={float(np.linalg.norm(w)):.3f}")
        except Exception:
            pass
        return _logit, "Logit"
    except Exception:
        return None, "logit_error"


def _build_xgb_scorer(
    vm_choice: str, lstate: Any, prompt: str, session_state: Any
) -> Tuple[Callable[[np.ndarray], float], str]:
    """Simplified XGB scorer: no session cache.

    We return unavailable unless a live XGB model is explicitly provided by callers
    in session (optional `XGB_MODEL`), keeping logic simple and predictable.
    """
    def _zero(_fvec: np.ndarray) -> float:
        return 0.0

    try:
        mdl = _get_live_xgb_model(session_state)
        if mdl is None:
            _print_xgb_unavailable(vm_choice, lstate, prompt, session_state)
            return _zero, "xgb_unavailable"

        # Prefer top-level stub when present (tests), otherwise package path
        try:
            import sys as _sys
            if "xgb_value" in _sys.modules and hasattr(_sys.modules["xgb_value"], "score_xgb_proba"):
                from xgb_value import score_xgb_proba  # type: ignore
            else:
                from ipo.core.xgb_value import score_xgb_proba  # type: ignore
        except Exception:  # final fallback
            from ipo.core.xgb_value import score_xgb_proba  # type: ignore

        def _xgb(fvec: np.ndarray) -> float:
            return float(score_xgb_proba(mdl, np.asarray(fvec, dtype=float)))

        return _xgb, "ok"
    except Exception:
        print("[xgb] scorer error; returning 0 for all scores")
        return _zero, "xgb_error"


def _get_live_xgb_model(session_state: Any):
    """Return a live XGB model from a single consolidated helper."""
    try:
        from ipo.core.xgb_value import get_live_model  # type: ignore
        return get_live_model(session_state)
    except Exception:
        return None


def _print_xgb_unavailable(vm_choice: str, lstate: Any, prompt: str, session_state: Any) -> None:
    """Emit the same unavailable line as before (keeps tests stable)."""
    try:
        from ipo.core.persistence import get_dataset_for_prompt_or_session as _get_ds  # type: ignore

        Xd, _ = _get_ds(prompt, session_state)
        rows = int(getattr(Xd, "shape", (0, 0))[0]) if Xd is not None else 0
        d_lat = getattr(lstate, "d", "?")
        print(
            f"[xgb] scorer unavailable: no model (vm={vm_choice}, dataset_rows={rows}, d={d_lat})"
        )
    except Exception:
        print("[xgb] scorer unavailable: no model")


## Legacy non‑ridge value models were removed.


def get_value_scorer(
    vm_choice: str, lstate: Any, prompt: str, session_state: Any
) -> Tuple[Callable[[np.ndarray], float] | None, str]:
    """Unified scorer API.

    Returns (scorer|None, tag_or_status):
    - When a scorer is usable, tag is one of: "Ridge" | "XGB" | "Distance".
    - Otherwise returns (None, status) where status ∈ {"ridge_untrained","xgb_unavailable","xgb_training","xgb_error"}.
    """
    choice = str(vm_choice or "Ridge")
    if choice == "Distance":
        return _build_distance_scorer(session_state)

    if choice == "XGBoost":
        s, status = _build_xgb_scorer(choice, lstate, prompt, session_state)
        return (s if status == "ok" else None), ("XGB" if status == "ok" else status)

    if choice == "Logistic":
        s, status = _build_logit_scorer(session_state)
        return (s if status == "Logit" else None), status

    # Ridge (default)
    s, status = _build_ridge_scorer(lstate)
    return (s if status == "ok" else None), ("Ridge" if status == "ok" else status)


def get_value_scorer_with_status(
    vm_choice: str, lstate: Any, prompt: str, session_state: Any
) -> Tuple[Callable[[np.ndarray], float], str]:
    """Backwards-compat shim: maps unified API to (scorer, status)."""
    scorer, tag_or_status = get_value_scorer(vm_choice, lstate, prompt, session_state)
    if scorer is None:
        # When unavailable, return a zero scorer with the status string
        def _zero(_fvec: np.ndarray) -> float:
            return 0.0

        return _zero, tag_or_status
    # tag maps to ok
    return scorer, "ok"
