from __future__ import annotations

import time as _time
from datetime import datetime, timezone
import logging as _logging
from typing import Any

import numpy as np
from ipo.infra.constants import Keys

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


def _fit_ridge(lstate: Any, X: np.ndarray, y: np.ndarray, lam: float) -> None:
    """Fit ridge weights synchronously and store into lstate.w."""
    try:
        from latent_logic import ridge_fit  # local import keeps import time low

        w_new = ridge_fit(X, y, float(lam))
        lock = getattr(lstate, "w_lock", None)
        if lock is not None:
            with lock:
                lstate.w = w_new
        else:
            lstate.w = w_new
        try:
            nrm = float(np.linalg.norm(w_new[: getattr(lstate, "d", w_new.shape[0])]))
            _log(f"[ridge] fit rows={X.shape[0]} d={X.shape[1]} lam={lam} ||w||={nrm:.3f}")
        except Exception:
            pass
    except Exception:
        pass


def _maybe_fit_xgb(X: np.ndarray, y: np.ndarray, lam: float, session_state: Any) -> None:
    """Sync XGB fit with minimal side effects; also updates legacy cache for compat."""
    try:
        from ipo.core.xgb_value import fit_xgb_classifier  # type: ignore

        n = int(X.shape[0])
        d = int(X.shape[1]) if X.ndim == 2 else 0
        if n <= 0 or not _has_two_classes(y):
            classes = set(np.asarray(y).astype(int).tolist()) if n > 0 else set()
            _log(
                f"[xgb] skip: insufficient classes rows={n} classes={sorted(list(classes)) if classes else []}"
            )
            return
        yy = np.asarray(y).astype(int)
        pos = int((yy > 0).sum())
        neg = int((yy < 0).sum())
        # Clear stale future handles (sync-only path)
        try:
            session_state.pop(Keys.XGB_FIT_FUTURE, None)
        except Exception:
            pass
        n_estim, max_depth = _xgb_hparams(session_state)
        _log(f"[xgb] train start rows={n} d={d} pos={pos} neg={neg}")
        _log(f"[xgb] params n_estim={n_estim} depth={max_depth}")
        t_x = _time.perf_counter()
        mdl = fit_xgb_classifier(X, y, n_estimators=n_estim, max_depth=max_depth)
        _store_xgb_model(session_state, mdl, n)
        dt_ms = (_time.perf_counter() - t_x) * 1000.0
        _log(f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms")
        try:
            session_state[Keys.XGB_TRAIN_STATUS] = {"state": "ok", "rows": int(n), "lam": float(lam)}
        except Exception:
            pass
    except Exception:
        pass


def _xgb_hparams(session_state: Any) -> tuple[int, int]:
    try:
        n_estim = int(session_state.get("xgb_n_estimators", 50))
    except Exception:
        n_estim = 50
    try:
        max_depth = int(session_state.get("xgb_max_depth", 3))
    except Exception:
        max_depth = 3
    return n_estim, max_depth


def _store_xgb_model(session_state: Any, mdl: Any, n_rows: int) -> None:
    try:
        session_state.XGB_MODEL = mdl
        cache = getattr(session_state, "xgb_cache", {}) or {}
        cache["model"] = mdl
        cache["n"] = int(n_rows)
        session_state.xgb_cache = cache
        session_state["xgb_toast_ready"] = True
    except Exception:
        pass


def _has_two_classes(y: np.ndarray) -> bool:
    try:
        classes = set(np.asarray(y).astype(int).tolist())
        return len(classes) > 1
    except Exception:
        return False


def _maybe_fit_logit(X: np.ndarray, y: np.ndarray, lam: float, session_state: Any) -> None:
    """Tiny numpy-only logistic regression (sync)."""
    try:
        n = int(X.shape[0])
        d = int(X.shape[1]) if X.ndim == 2 else 0
        if n <= 0 or d <= 0:
            _log("[logit] skip: empty dataset")
            return
        yy = np.asarray(y).astype(float)
        y01 = (yy > 0).astype(float)
        W = np.asarray(session_state.get(Keys.LOGIT_W) or np.zeros(d), dtype=float)
        steps, lam_eff = _logit_params(session_state, lam)
        W = _logit_train_loop(X, y01, W, n, steps, lam_eff)
        session_state[Keys.LOGIT_W] = W
        _log(f"[logit] fit rows={n} d={d} steps={steps} lam={lam_eff} ||w||={float(np.linalg.norm(W)):.3f}")
    except Exception:
        pass


def _logit_params(session_state: Any, lam_fallback: float) -> tuple[int, float]:
    try:
        steps = int(session_state.get(Keys.LOGIT_STEPS) or 120)
    except Exception:
        steps = 120
    try:
        lam_eff = float(session_state.get(Keys.LOGIT_L2)) if session_state.get(Keys.LOGIT_L2) is not None else float(lam_fallback)
    except Exception:
        lam_eff = float(lam_fallback)
    return steps, lam_eff


def _logit_train_loop(X: np.ndarray, y01: np.ndarray, W: np.ndarray, n: int, steps: int, lam_eff: float) -> np.ndarray:
    lr = 0.1
    for _ in range(int(steps)):
        z = X @ W
        p = 1.0 / (1.0 + np.exp(-z))
        g = (X.T @ (p - y01)) / float(n) + float(lam_eff) * W
        W = W - lr * g
    return W


def _train_optionals(vm_choice: str, lstate: Any, X: np.ndarray, y: np.ndarray, lam: float, session_state: Any) -> None:
    choice = str(vm_choice)
    if choice == "XGBoost":
        _maybe_fit_xgb(X, y, float(lam), session_state)
    elif choice == "Logistic":
        _maybe_fit_logit(X, y, float(lam), session_state)


def _ridge_summary(lstate: Any, X: np.ndarray, yy: np.ndarray, lam: float) -> None:
    try:
        n = int(X.shape[0])
        d = int(X.shape[1]) if X.ndim == 2 else 0
        wv = getattr(lstate, "w", None)
        if wv is not None and n > 0:
            yhat = (np.dot(X, wv) >= 0.0)
            acc = float((yhat == (yy > 0)).mean())
            pos = int((yy > 0).sum())
            neg = int((yy < 0).sum())
            _log(f"[train-summary] ridge rows={n} d={d} lam={lam} acc={acc*100:.0f}% pos={pos} neg={neg}")
    except Exception:
        pass


def _logit_summary(X: np.ndarray, yy: np.ndarray, lam: float, session_state: Any) -> None:
    try:
        from ipo.infra.constants import Keys as _K

        n = int(X.shape[0])
        d = int(X.shape[1]) if X.ndim == 2 else 0
        W = session_state.get(_K.LOGIT_W)
        if W is not None and n > 0:
            z = np.dot(X, np.asarray(W, dtype=float))
            p = 1.0 / (1.0 + np.exp(-z))
            yhat = (p >= 0.5)
            acc = float((yhat == (yy > 0)).mean())
            steps = int(session_state.get(_K.LOGIT_STEPS) or 0)
            lam_eff = float(session_state.get(_K.LOGIT_L2) or lam)
            pos = int((yy > 0).sum())
            neg = int((yy < 0).sum())
            _log(f"[train-summary] logit rows={n} d={d} steps={steps} lam={lam_eff} acc={acc*100:.0f}% pos={pos} neg={neg}")
    except Exception:
        pass


def _xgb_summary(X: np.ndarray, yy: np.ndarray, session_state: Any) -> None:
    try:
        mdl = getattr(session_state, "XGB_MODEL", None)
        if mdl is not None and int(X.shape[0]) > 0:
            from ipo.core.xgb_value import score_xgb_proba  # type: ignore

            p = np.asarray([score_xgb_proba(mdl, x) for x in X], dtype=float)
            yhat = (p >= 0.5)
            acc = float((yhat == (yy > 0)).mean())
            n = int(X.shape[0])
            d = int(X.shape[1]) if X.ndim == 2 else 0
            pos = int((yy > 0).sum())
            neg = int((yy < 0).sum())
            _log(f"[train-summary] xgb rows={n} d={d} acc={acc*100:.0f}% pos={pos} neg={neg}")
    except Exception:
        pass


def _record_train_summaries(lstate: Any, X: np.ndarray, y: np.ndarray, lam: float, session_state: Any) -> None:
    try:
        yy = np.asarray(y).astype(float)
        _ridge_summary(lstate, X, yy, lam)
        _logit_summary(X, yy, lam, session_state)
        _xgb_summary(X, yy, session_state)
    except Exception:
        pass


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

    # Optional XGB/Logistic refresh via helpers
    _train_optionals(choice, lstate, X, y, lam, session_state)

    # Update ridge weights for w only when Ridge-like modes are active.
    if _uses_ridge(choice):
        _fit_ridge(lstate, X, y, float(lam))

    # Training bookkeeping
    try:
        session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    except Exception:
        pass

    # Print compact train summaries to CLI
    _record_train_summaries(lstate, X, y, lam, session_state)


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
