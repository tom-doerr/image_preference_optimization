from datetime import datetime, timezone

import numpy as np

from ipo.infra.constants import Keys


class XGBTrainer:
    def __init__(self, n_estimators=50, max_depth=8):
        self.n_estimators, self.max_depth = int(n_estimators), int(max_depth)
    def fit(self, X, y, warm_model=None):
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=0.1)  # noqa: E501
        m.fit(X, y, xgb_model=warm_model)
        return m

def _xgb_proba(model, fvec):
    import time
    t0 = time.time()
    p = float(model.predict_proba(np.asarray(fvec).reshape(1, -1))[0, 1])
    dt = time.time() - t0
    if dt > 0.05:  # log if >50ms
        print(f"[xgb_proba] {dt*1000:.0f}ms")
    return p

def _get_xgb_params(ss):
    try:
        if hasattr(ss, "get"):
            n = ss.get(Keys.XGB_N_ESTIMATORS, 50)
            d = ss.get(Keys.XGB_MAX_DEPTH, 8)
        else:
            n = getattr(ss, Keys.XGB_N_ESTIMATORS, 50)
            d = getattr(ss, Keys.XGB_MAX_DEPTH, 8)
        return int(n or 50), int(d or 8)
    except (AttributeError, KeyError, TypeError, ValueError):
        return 50, 8

def _set_xgb_model(ss, model, n):
    try:
        ss.XGB_MODEL = model
        ss.xgb_cache = {"model": model, "n": n}
    except (AttributeError, TypeError):
        pass

def _get_xgb_model(ss):
    return getattr(ss, "XGB_MODEL", None) or (getattr(ss, "xgb_cache", {}) or {}).get("model")

def _fit_ridge(lstate, X, y, lam):
    from ipo.core.latent_state import ridge_fit
    w = ridge_fit(X, y, float(lam))
    lock = getattr(lstate, "w_lock", None)
    if lock:
        with lock:
            lstate.w = w
    else:
        lstate.w = w


def _has_two_classes(y): return len(set(np.asarray(y).astype(int).tolist())) > 1

def _fit_gaussian(X, y, ss):
    """Fit per-dim Gaussian from good samples."""
    mask = np.asarray(y) > 0
    if mask.sum() < 2:
        return
    Xg = X[mask]
    mu, sigma = Xg.mean(axis=0), Xg.std(axis=0) + 1e-6
    ss["gauss_mu"], ss["gauss_sigma"] = mu, sigma

def _gauss_logp(mu, sigma, z):
    """Log prob under diagonal Gaussian (unnormalized)."""
    return -0.5 * np.sum(((z - mu) / sigma) ** 2)

def _maybe_fit_xgb(X, y, lam, ss):
    import time
    if X.shape[0] <= 0 or not _has_two_classes(y):
        return
    n_estim, max_depth = _get_xgb_params(ss)
    y01 = ((np.asarray(y) + 1) / 2).astype(int)  # -1,1 -> 0,1
    old = _get_xgb_model(ss)
    t0 = time.time()
    mdl = XGBTrainer(n_estim, max_depth).fit(X, y01, warm_model=old)
    print(f"[xgb] {'warm' if old else 'cold'} fit {X.shape[0]} in {time.time()-t0:.2f}s")
    _set_xgb_model(ss, mdl, X.shape[0])

def _train_optionals(vm_choice, lstate, X, y, lam, ss):
    if str(vm_choice) == "XGBoost":
        _maybe_fit_xgb(X, y, float(lam), ss)
    if str(vm_choice) == "Gaussian":
        _fit_gaussian(X, y, ss)

def fit_value_model(vm_choice, lstate, X, y, lam, session_state):
    _train_optionals(str(vm_choice), lstate, X, y, lam, session_state)
    mode = session_state.get(Keys.XGB_OPTIM_MODE) if hasattr(session_state, "get") else None
    if str(vm_choice) == "XGBoost" and mode == "Line":
        print("[train] XGB done, training Ridge for line search direction")
    _fit_ridge(lstate, X, y, float(lam))
    try:
        session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    except (AttributeError, TypeError, KeyError):
        pass


def get_vm(choice): return choice

def ensure_fitted(vm_choice, lstate, X, y, lam, session_state):
    try:
        if X is None or int(X.shape[0]) <= 0:
            return
        fit_value_model(vm_choice, lstate, X, y, float(lam), session_state)
    except (ValueError, AttributeError, TypeError):
        pass

def get_value_scorer(vm_choice, lstate, prompt, ss):
    c = str(vm_choice or "Ridge")
    if c == "XGBoost":
        mdl = _get_xgb_model(ss)
        if mdl:
            return (lambda f: _xgb_proba(mdl, f)), "XGB"
        return None, "xgb_unavailable"
    if c == "Gaussian":
        mu, sig = ss.get("gauss_mu"), ss.get("gauss_sigma")
        if mu is not None:
            return (lambda f: _gauss_logp(mu, sig, f)), "Gauss"
        return None, "gauss_unavailable"
    w = getattr(lstate, "w", None)
    if w is None:
        return None, "untrained"
    ww = np.asarray(w[:int(getattr(lstate, "d", len(w)))], dtype=float)
    return (lambda f: float(np.dot(ww, f))), "Ridge"
