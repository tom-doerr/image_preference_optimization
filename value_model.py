from __future__ import annotations

import time as _time
from datetime import datetime, timezone
from typing import Any

import numpy as np

__all__ = [
    'fit_value_model',
    'ensure_fitted',
]


def _uses_ridge(choice: str) -> bool:
    """Return True when the selected value model should fit Ridge w."""
    c = str(choice)
    return c not in ('DistanceHill', 'CosineHill')


def fit_value_model(
    vm_choice: str,
    lstate: Any,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    session_state: Any,
) -> None:
    """Fit/update the value model artifacts with minimal logic.

    - Always fits Ridge to update lstate.w (keeps proposals simple and fast).
    - If vm_choice == 'XGBoost', also (re)fit and cache an XGB model when
      row count changes and both classes are present. Scores are consumed via
      value_scorer.get_value_scorer.
    - Records last_train_at and last_train_ms in session_state.
    """
    t0 = _time.perf_counter()
    try:
        print(f"[train] start vm={vm_choice} rows={X.shape[0]} d={X.shape[1]} lam={lam}")
    except Exception:
        pass

    # Optional XGB cache refresh
    choice = str(vm_choice)
    if choice == 'XGBoost':
        try:
            from xgb_value import fit_xgb_classifier  # type: ignore
            n = int(X.shape[0])
            d = int(X.shape[1]) if X.ndim == 2 else 0
            if n > 0 and len(set(np.asarray(y).astype(int).tolist())) > 1:
                yy = np.asarray(y).astype(int)
                pos = int((yy > 0).sum())
                neg = int((yy < 0).sum())
                try:
                    print(f"[xgb] train start rows={n} d={d} pos={pos} neg={neg}")
                except Exception:
                    pass
                cache = getattr(session_state, 'xgb_cache', {}) or {}
                last_n = int(cache.get('n') or 0)
                # Read simple hyperparams from session_state; default to 50/3.
                try:
                    n_estim = int(getattr(session_state, "xgb_n_estimators", session_state.get("xgb_n_estimators", 50)))
                except Exception:
                    n_estim = 50
                try:
                    max_depth = int(getattr(session_state, "xgb_max_depth", session_state.get("xgb_max_depth", 3)))
                except Exception:
                    max_depth = 3
                do_async = bool(getattr(session_state, "xgb_train_async", False))
                if cache.get('model') is None or last_n != n:
                    if do_async:
                        try:
                            from background import get_executor  # lazy import

                            def _fit_and_store():
                                t_x = _time.perf_counter()
                                mdl_inner = fit_xgb_classifier(X, y, n_estimators=n_estim, max_depth=max_depth)
                                session_state.xgb_cache = {'model': mdl_inner, 'n': n}
                                try:
                                    dt_ms = (_time.perf_counter() - t_x) * 1000.0
                                    print(f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms (async)")
                                except Exception:
                                    pass
                                try:
                                    session_state["xgb_train_status"] = {"state": "ok", "rows": n, "lam": float(lam)}
                                    session_state["xgb_last_updated_rows"] = int(n)
                                    session_state["xgb_last_updated_lam"] = float(lam)
                                except Exception:
                                    pass
                                return mdl_inner

                            session_state["xgb_train_status"] = {"state": "running", "rows": n, "lam": float(lam)}
                            fut = get_executor().submit(_fit_and_store)
                            session_state["xgb_fit_future"] = fut
                        except Exception:
                            # Fallback to sync if executor fails
                            t_x = _time.perf_counter()
                            mdl = fit_xgb_classifier(X, y, n_estimators=n_estim, max_depth=max_depth)
                            session_state.xgb_cache = {'model': mdl, 'n': n}
                            try:
                                dt_ms = (_time.perf_counter() - t_x) * 1000.0
                                print(f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms")
                            except Exception:
                                pass
                    else:
                        t_x = _time.perf_counter()
                        mdl = fit_xgb_classifier(X, y, n_estimators=n_estim, max_depth=max_depth)
                        session_state.xgb_cache = {'model': mdl, 'n': n}
                        try:
                            dt_ms = (_time.perf_counter() - t_x) * 1000.0
                            print(f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms")
                        except Exception:
                            pass
        except Exception:
            pass

    # Update ridge weights for w only when Ridge-like modes are active.
    # DistanceHill/CosineHill rely on distance/cosine scorers over the dataset.
    if _uses_ridge(choice):
        try:
            from latent_logic import ridge_fit  # local import keeps import time low
            w_new = ridge_fit(X, y, float(lam))
            lstate.w = w_new
            try:
                nrm = float(np.linalg.norm(w_new[: getattr(lstate, "d", w_new.shape[0])]))
                print(f"[ridge] fit rows={X.shape[0]} d={X.shape[1]} lam={lam} ||w||={nrm:.3f}")
            except Exception:
                pass
        except Exception:
            pass

    # Training bookkeeping
    try:
        session_state['last_train_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    except Exception:
        pass
    try:
        session_state['last_train_ms'] = float((_time.perf_counter() - t0) * 1000.0)
        print(f"[perf] train: rows={X.shape[0]} d={X.shape[1]} took {session_state['last_train_ms']:.1f} ms")
    except Exception:
        pass
    # Mark async fit as done if we were running in background
    try:
        session_state["xgb_train_status"] = {"state": "ok", "rows": int(X.shape[0]), "lam": float(lam)}
        session_state["xgb_last_updated_rows"] = int(X.shape[0])
        session_state["xgb_last_updated_lam"] = float(lam)
    except Exception:
        pass
    try:
        fut = session_state.get("xgb_fit_future")
        if fut is not None and hasattr(fut, "done"):
            fut._done = True  # simple flag; don't rely on Future internals
    except Exception:
        pass


def ensure_fitted(
    vm_choice: str,
    lstate: Any,
    X: Any,
    y: Any,
    lam: float,
    session_state: Any,
) -> None:
    """Lazy-fit Ridge/XGBoost when a usable dataset exists and no model is ready.

    - Requires X,y to be non-empty and feature dim to match lstate.d.
    - If ‖w‖≈0 and there is no XGB cache yet, calls fit_value_model once and
      records a small guard flag in session_state['auto_fit_done'].
    """
    try:
        import numpy as _np
        if X is None or y is None or getattr(X, "shape", (0,))[0] == 0:
            return
        # Dimensionality check
        try:
            d_x = int(getattr(X, "shape", (0, 0))[1])
            d_lat = int(getattr(lstate, "d", d_x))
            if d_x != d_lat:
                return
        except Exception:
            return
        w_now = getattr(lstate, "w", None)
        w_norm = float(_np.linalg.norm(w_now)) if w_now is not None else 0.0
    except Exception:
        return
    cache = getattr(session_state, "xgb_cache", {}) or {}
    auto_flag = getattr(session_state, "auto_fit_done", False) or bool(session_state.get("auto_fit_done", False))
    # For Ridge-like modes we trigger once when w is still zero. For XGBoost
    # we also auto-fit when there is no cached model yet, even if w was
    # restored from a previous session, so the XGB scorer becomes available.
    needs_xgb = str(vm_choice) == "XGBoost"
    if ((w_norm == 0.0) or needs_xgb) and not cache and not auto_flag:
        fit_value_model(vm_choice, lstate, X, y, lam, session_state)
        try:
            session_state["auto_fit_done"] = True
        except Exception:
            pass
