def fit_xgb_classifier(X, y):
    # Minimal wrapper to avoid leaking xgboost details elsewhere
    import xgboost as xgb  # noqa: F401
    # Map labels {-1,1} -> {0,1}
    import numpy as np
    yb = ((y > 0).astype(int)).ravel()
    # Lightweight defaults; CPU; deterministic enough for tiny datasets
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=1,
        tree_method='hist',
        verbosity=0,
    )
    model.fit(X, yb)
    return model


def score_xgb_proba(model, fvec):
    import numpy as np
    proba = model.predict_proba(np.asarray(fvec, dtype=float).reshape(1, -1))[0, 1]
    return float(proba)

