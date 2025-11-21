from __future__ import annotations

import time as _time
from datetime import datetime, timezone
import logging as _logging
from typing import Any

import numpy as np
from constants import Keys

__all__ = [
    "fit_value_model",
    "ensure_fitted",  # compat shim; sync-only
]

LOGGER = _logging.getLogger("ipo")
if not LOGGER.handlers:
    try:
        _h = _logging.FileHandler("ipo.debug.log")
        _h.setFormatter(
            _logging.Formatter("%(asctime)s %(levelname)s value_model: %(message)s")
        )
        LOGGER.addHandler(_h)
        LOGGER.setLevel(_logging.INFO)
    except Exception:
        pass
try:
    import os as _os

    _lvl = (_os.getenv("IPO_LOG_LEVEL") or "").upper()
    if _lvl:
        LOGGER.setLevel(getattr(_logging, _lvl, _logging.INFO))
except Exception:
    pass


def _log(msg: str, level: str = "info") -> None:
    """Log to stdout (tests) and ipo logger."""
    try:
        print(msg)
    except Exception:
        pass
    try:
        getattr(LOGGER, level, LOGGER.info)(msg)
    except Exception:
        pass


# Per-state lock lives on LatentState (lstate.w_lock). Keep no global lock.




def _uses_ridge(choice: str) -> bool:
    """All supported modes train Ridge weights (DH/CH pruned)."""
    return True


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
    choice = str(vm_choice)

    # Ridge fits are always synchronous now; ignore any async toggles.
    _log(f"[train] start vm={vm_choice} rows={X.shape[0]} d={X.shape[1]} lam={lam}")

    # Optional XGB cache refresh
    # choice already set above
    if choice == "XGBoost":
        try:
            from xgb_value import fit_xgb_classifier  # type: ignore

            n = int(X.shape[0])
            d = int(X.shape[1]) if X.ndim == 2 else 0
            classes = set(np.asarray(y).astype(int).tolist()) if n > 0 else set()
            if n > 0 and len(classes) > 1:
                yy = np.asarray(y).astype(int)
                pos = int((yy > 0).sum())
                neg = int((yy < 0).sum())
                cache = getattr(session_state, Keys.XGB_CACHE, {}) or {}
                last_n = int(cache.get("n") or 0)
                # 195a: no async future; ignore any stale future handle and fit synchronously
                # Clear any stale future handle; sync-only now
                try:
                    session_state.pop(Keys.XGB_FIT_FUTURE, None)
                except Exception:
                    pass
                # Read simple hyperparams from session_state; default to 50/3.
                try:
                    n_estim = int(
                        getattr(
                            session_state,
                            "xgb_n_estimators",
                            session_state.get("xgb_n_estimators", 50),
                        )
                    )
                except Exception:
                    n_estim = 50
                try:
                    max_depth = int(
                        getattr(
                            session_state,
                            "xgb_max_depth",
                            session_state.get("xgb_max_depth", 3),
                        )
                    )
                except Exception:
                    max_depth = 3
                if cache.get("model") is None or last_n != n:
                    _log(f"[xgb] train start rows={n} d={d} pos={pos} neg={neg}")
                    try:
                        _log(f"[xgb] params n_estim={n_estim} depth={max_depth}")
                    except Exception:
                        pass
                    t_x = _time.perf_counter()
                    mdl = fit_xgb_classifier(X, y, n_estimators=n_estim, max_depth=max_depth)
                    session_state.xgb_cache = {"model": mdl, "n": n}
                    try:
                        session_state["xgb_toast_ready"] = True
                    except Exception:
                        pass
                    dt_ms = (_time.perf_counter() - t_x) * 1000.0
                    _log(f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms")
                    try:
                        session_state[Keys.XGB_TRAIN_STATUS] = {
                            "state": "ok",
                            "rows": int(n),
                            "lam": float(lam),
                        }
                    except Exception:
                        pass
                else:
                    _log(f"[xgb] skip: cache up-to-date rows={n}")
            else:
                _log(f"[xgb] skip: insufficient classes rows={n} classes={sorted(list(classes)) if classes else []}")
        except Exception:
            pass

    # Update ridge weights for w only when Ridge-like modes are active.
    # Non‑ridge legacy modes are removed; Ridge weights always trained.
    if _uses_ridge(choice):
        try:
            from latent_logic import ridge_fit  # local import keeps import time low

            # 199a: Ridge fits are synchronous only
            w_new = ridge_fit(X, y, float(lam))
            lock = getattr(lstate, "w_lock", None)
            if lock is not None:
                with lock:
                    lstate.w = w_new
            else:
                lstate.w = w_new
            try:
                nrm = float(np.linalg.norm(w_new[: getattr(lstate, "d", w_new.shape[0])]))
                _log(
                    f"[ridge] fit rows={X.shape[0]} d={X.shape[1]} lam={lam} ||w||={nrm:.3f}"
                )
            except Exception:
                pass
        except Exception:
            pass

    # Training bookkeeping
    try:
        session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    except Exception:
        pass


def ensure_fitted(
    vm_choice: str,
    lstate: Any,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    session_state: Any,
) -> None:
    """Compatibility shim: perform a sync fit if data looks usable.

    - For XGBoost, require at least one positive and one negative label.
    - Always refresh Ridge weights via fit_value_model (keeps code centralized).
    - Records LAST_TRAIN_AT/MS via fit_value_model.
    """
    try:
        n = int(getattr(X, "shape", (0,))[0]) if X is not None else 0
        if n <= 0:
            return
        if str(vm_choice) == "XGBoost":
            yy = np.asarray(y).astype(int) if y is not None else np.zeros(0, dtype=int)
            if len(set(yy.tolist())) <= 1:
                # not enough classes; only refresh Ridge
                _log("[ensure] ridge sync fit (xgb insufficient classes)")
                return fit_value_model("Ridge", lstate, X, y, float(lam), session_state)
            _log("[ensure] xgb sync fit")
        # Delegate to main trainer (sync)
        if str(vm_choice) == "Ridge":
            _log("[ensure] ridge sync fit")
        return fit_value_model(vm_choice, lstate, X, y, float(lam), session_state)
    except Exception:
        # Minimal shim; swallow to keep import-time behavior stable
        return


def _ensure_fitted_removed(
    vm_choice: str,
    lstate: Any,
    X: Any,
    y: Any,
    lam: float,
    session_state: Any,
) -> None:
    """199c: ensure_fitted retired; do nothing (kept for import compatibility)."""
    try:
        session_state["auto_fit_done"] = True
    except Exception:
        pass


def _train_and_record_removed(
    vm_choice: str,
    lstate: Any,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    session_state: Any,
) -> str:
    """199c: train_and_record retired; call fit_value_model directly in UI."""
    fit_value_model(vm_choice, lstate, X, y, lam, session_state)
    return "ok"
