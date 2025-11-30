"""Hyperparameter search for XGBoost using Optuna."""
import numpy as np


def xgb_hparam_search(X, y, n_trials=100, cv=3):
    """Optuna-based XGB hyperparameter search."""
    import optuna
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    y01 = ((np.asarray(y) + 1) / 2).astype(int)
    tscv = TimeSeriesSplit(n_splits=cv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        n = trial.suggest_int("n_estimators", 10, 200)
        d = trial.suggest_int("max_depth", 2, 8)
        lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        reg = trial.suggest_float("reg_lambda", 0.01, 10.0, log=True)
        mcw = trial.suggest_int("min_child_weight", 1, 10)
        sub = trial.suggest_float("subsample", 0.5, 1.0)
        col = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        m = xgb.XGBClassifier(
            n_estimators=n, max_depth=d, learning_rate=lr, reg_lambda=reg,
            min_child_weight=mcw, subsample=sub, colsample_bytree=col,
            verbosity=0, n_jobs=1)
        return cross_val_score(m, X, y01, cv=tscv).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    best_p = {
        "n": bp["n_estimators"], "d": bp["max_depth"],
        "lr": bp["learning_rate"], "reg": bp["reg_lambda"],
        "mcw": bp["min_child_weight"], "sub": bp["subsample"],
        "col": bp["colsample_bytree"]}
    print(f"[hp] best={best_p} cv={study.best_value:.3f}")
    return best_p, study.best_value
