from __future__ import annotations

import logging as _logging
import time as _time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ipo.infra.constants import Keys, SAFE_EXC

class XGBTrainer:
    def __init__(self, n_estimators=50, max_depth=3):
        self.n_estimators, self.max_depth = int(n_estimators), int(max_depth)
    def fit(self, X, y):
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=0.1)  # noqa: E501
        m.fit(X, y); return m

def _xgb_proba(model, fvec):
    return float(model.predict_proba(np.asarray(fvec).reshape(1, -1))[0, 1])

def _get_xgb_params(ss):
    try: return int(getattr(ss, "xgb_n_estimators", 50)), int(getattr(ss, "xgb_max_depth", 3))
    except Exception: return 50, 3

def _set_xgb_model(ss, model, n):
    try: ss.XGB_MODEL = model; ss.xgb_cache = {"model": model, "n": n}
    except Exception: pass

def _get_xgb_model(ss):
    return getattr(ss, "XGB_MODEL", None) or (getattr(ss, "xgb_cache", {}) or {}).get("model")

__all__ = ["fit_value_model", "ensure_fitted", "get_value_scorer", "XGBTrainer", "get_vm"]

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
    except SAFE_EXC:
        pass
    try:
        getattr(LOGGER, level, LOGGER.info)(msg)
    except SAFE_EXC:
        pass


# Per-state lock lives on LatentState (lstate.w_lock). Keep no global lock.




def _uses_ridge(choice: str) -> bool:
    """All supported modes train Ridge weights (DH/CH pruned)."""
    return True


def _fit_ridge(lstate: Any, X: np.ndarray, y: np.ndarray, lam: float) -> None:
    """Fit ridge weights synchronously and store into lstate.w."""
    try:
        from ipo.core.latent_state import ridge_fit  # local import

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
    """Sync XGB fit."""
    try:
        n = int(X.shape[0])
        d = int(X.shape[1]) if X.ndim == 2 else 0
        if n <= 0 or not _has_two_classes(y):
            classes = set(np.asarray(y).astype(int).tolist()) if n > 0 else set()
            _log(f"[xgb] skip: insufficient classes rows={n} classes={sorted(list(classes)) if classes else []}")  # noqa: E501
            return
        yy = np.asarray(y).astype(int)
        pos = int((yy > 0).sum())
        neg = int((yy < 0).sum())
        # Async futures removed; no future bookkeeping
        n_estim, max_depth = _get_xgb_params(session_state)
        _log(f"[xgb] train rows={n} pos={pos} neg={neg}")
        t_x = _time.perf_counter()
        mdl = XGBTrainer(n_estim, max_depth).fit(X, y)
        _set_xgb_model(session_state, mdl, n)
        dt_ms = (_time.perf_counter() - t_x) * 1000.0
        _log(f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms")
        try:
            session_state[Keys.XGB_TRAIN_STATUS] = {"state": "ok", "rows": int(n), "lam": float(lam)}  # noqa: E501
        except Exception:
            pass
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
        _log(f"[logit] fit rows={n} d={d} steps={steps} lam={lam_eff} ||w||={float(np.linalg.norm(W)):.3f}")  # noqa: E501
    except Exception:
        pass


def _logit_params(session_state: Any, lam_fallback: float) -> tuple[int, float]:
    try:
        steps = int(session_state.get(Keys.LOGIT_STEPS) or 120)
    except Exception:
        steps = 120
    try:
        lam_eff = float(session_state.get(Keys.LOGIT_L2)) if session_state.get(Keys.LOGIT_L2) is not None else float(lam_fallback)  # noqa: E501
    except Exception:
        lam_eff = float(lam_fallback)
    return steps, lam_eff


def _logit_train_loop(X: np.ndarray, y01: np.ndarray, W: np.ndarray, n: int, steps: int, lam_eff: float) -> np.ndarray:  # noqa: E501
    lr = 0.1
    for _ in range(int(steps)):
        z = X @ W
        p = 1.0 / (1.0 + np.exp(-z))
        g = (X.T @ (p - y01)) / float(n) + float(lam_eff) * W
        W = W - lr * g
    return W


def _train_optionals(vm_choice: str, lstate: Any, X: np.ndarray, y: np.ndarray, lam: float, session_state: Any) -> None:  # noqa: E501
    choice = str(vm_choice)
    if choice == "XGBoost":
        _maybe_fit_xgb(X, y, float(lam), session_state)
    elif choice == "Logistic":
        _maybe_fit_logit(X, y, float(lam), session_state)




def fit_value_model(vm_choice, lstate, X, y, lam, session_state):
    _train_optionals(str(vm_choice), lstate, X, y, lam, session_state)
    _fit_ridge(lstate, X, y, float(lam))
    try: session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception: pass


def get_vm(choice): return choice

def ensure_fitted(vm_choice, lstate, X, y, lam, session_state):
    try:
        if X is None or int(X.shape[0]) <= 0: return
        fit_value_model(vm_choice, lstate, X, y, float(lam), session_state)
    except Exception: pass

def get_value_scorer(vm_choice, lstate, prompt, ss):
    c = str(vm_choice or "Ridge")
    if c == "XGBoost":
        mdl = _get_xgb_model(ss)
        if mdl: return (lambda f: _xgb_proba(mdl, f)), "XGB"
        return None, "xgb_unavailable"
    w = getattr(lstate, "w", None)
    if w is None: return None, "untrained"
    ww = np.asarray(w[:int(getattr(lstate, "d", len(w)))], dtype=float)
    return (lambda f: float(np.dot(ww, f))), "Ridge"
