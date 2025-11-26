from datetime import datetime, timezone
import numpy as np
from ipo.infra.constants import Keys

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

def _fit_ridge(lstate, X, y, lam):
    from ipo.core.latent_state import ridge_fit
    w = ridge_fit(X, y, float(lam))
    lock = getattr(lstate, "w_lock", None)
    if lock:
        with lock: lstate.w = w
    else: lstate.w = w


def _has_two_classes(y): return len(set(np.asarray(y).astype(int).tolist())) > 1

def _maybe_fit_xgb(X, y, lam, ss):
    if X.shape[0] <= 0 or not _has_two_classes(y): return
    n_estim, max_depth = _get_xgb_params(ss)
    y01 = ((np.asarray(y) + 1) / 2).astype(int)  # -1,1 -> 0,1
    mdl = XGBTrainer(n_estim, max_depth).fit(X, y01)
    _set_xgb_model(ss, mdl, X.shape[0])

def _train_optionals(vm_choice, lstate, X, y, lam, ss):
    if str(vm_choice) == "XGBoost": _maybe_fit_xgb(X, y, float(lam), ss)

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
