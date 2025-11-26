import numpy as np

from ipo.infra.util import SAFE_EXC

__all__ = [
    "XGBTrainer",
    "fit_xgb_classifier",  # compat shim
    "score_xgb_proba",     # compat shim
    "get_params",
    "set_live_model",
    "get_live_model",
]


class XGBTrainer:
    """Tiny OO wrapper around XGBoost fit/score.

    Kept intentionally minimal; tests can stub xgboost import or methods.
    """

    def __init__(self, n_estimators: int = 50, max_depth: int = 3):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)

    @classmethod
    def from_session(cls, session_state) -> "XGBTrainer":
        ne, md = get_params(session_state)
        return cls(ne, md)

    def fit(self, X, y):
        import xgboost as xgb  # type: ignore

        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1e-3,
            objective="binary:logistic",
            n_jobs=2,
        )
        model.fit(X, y)
        return model

    @staticmethod
    def proba(model, fvec) -> float:
        fv = np.asarray(fvec, dtype=float).reshape(1, -1)
        proba = float(model.predict_proba(fv)[0, 1])
        nrm = float(np.linalg.norm(fv))
        try:
            print(f"[xgb] eval d={fv.shape[1]} ‖f‖={nrm:.3f} proba={proba:.4f}")
        except SAFE_EXC:
            pass
        return proba


def fit_xgb_classifier(X, y, n_estimators: int = 50, max_depth: int = 3):
    """Compat: call through the OO trainer."""
    return XGBTrainer(n_estimators=n_estimators, max_depth=max_depth).fit(X, y)


def score_xgb_proba(model, fvec):
    """Compat: call through the OO trainer static scorer."""
    return XGBTrainer.proba(model, fvec)


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
    except SAFE_EXC:
        pass


def get_live_model(session_state):
    """Return a live trained XGB model from session state or None."""
    try:
        mdl = getattr(session_state, "XGB_MODEL", None)
        if mdl is not None:
            return mdl
    except SAFE_EXC:
        pass
    try:
        cache = getattr(session_state, "xgb_cache", {}) or {}
        return cache.get("model")
    except SAFE_EXC:
        return None
