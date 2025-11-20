import numpy as np


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
