import numpy as np
from typing import Optional


def fit_xgb_classifier(X, y, n_estimators: int = 50, max_depth: int = 3):
    """Minimal XGBoost classifier wrapper."""
    import xgboost as xgb  # type: ignore

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1e-3,
        objective="binary:logistic",
        n_jobs=2,
    )
    model.fit(X, y)
    return model


def score_xgb_proba(model, fvec):
    """Return P(y=1) for a single feature vector."""
    fv = np.asarray(fvec, dtype=float).reshape(1, -1)
    proba = float(model.predict_proba(fv)[0, 1])
    nrm = float(np.linalg.norm(fv))
    try:
        print(f"[xgb] eval d={fv.shape[1]} ‖f‖={nrm:.3f} proba={proba:.4f}")
    except Exception:
        pass
    return proba


# Consolidated session helpers
def get_params(session_state) -> tuple[int, int]:
    """Return (n_estimators, max_depth) from session_state with small defaults."""
    try:
        ne = int(getattr(session_state, "xgb_n_estimators", 50))
    except Exception:
        ne = 50
    try:
        md = int(getattr(session_state, "xgb_max_depth", 3))
    except Exception:
        md = 3
    return ne, md


def set_live_model(session_state, model, n_rows: int) -> None:
    """Store the trained model in session_state (and legacy cache) in one place."""
    try:
        session_state.XGB_MODEL = model
        cache = getattr(session_state, "xgb_cache", {}) or {}
        cache["model"] = model
        cache["n"] = int(n_rows)
        session_state.xgb_cache = cache
        session_state["xgb_toast_ready"] = True
    except Exception:
        pass


def get_live_model(session_state):
    """Return a live trained XGB model from session state or None."""
    try:
        mdl = getattr(session_state, "XGB_MODEL", None)
        if mdl is not None:
            return mdl
    except Exception:
        pass
    try:
        cache = getattr(session_state, "xgb_cache", {}) or {}
        return cache.get("model")
    except Exception:
        return None
