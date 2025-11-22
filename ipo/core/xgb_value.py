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


# --- CLI-compatible helpers merged here for convenience ---
def _prompt_hash(prompt: str) -> str:
    import hashlib

    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def train_xgb_for_prompt(
    prompt: str,
    n_estimators: int = 50,
    max_depth: int = 3,
    save_model: bool = True,
    model_dir: str = ".",
) -> Optional[str]:
    """Train XGBoost on the folder dataset for the prompt (sync, minimal).

    Returns saved model path when save_model is True, else None.
    """
    from ipo.core.persistence import get_dataset_for_prompt_or_session

    X, y = get_dataset_for_prompt_or_session(prompt, type("SS", (), {})())
    if X is None or y is None or X.shape[0] == 0:
        raise ValueError("Dataset is empty; add samples first.")
    if len(set(np.asarray(y).astype(int))) < 2:
        raise ValueError("Need both classes to train XGBoost (labels +1 and -1).")

    model = fit_xgb_classifier(X, y, n_estimators=n_estimators, max_depth=max_depth)
    preds = model.predict_proba(X)[:, 1] >= 0.5
    acc = float(np.mean(preds == (np.asarray(y) > 0)))

    out_path = None
    if save_model:
        import os

        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, f"xgb_model_{_prompt_hash(prompt)}.bin")
        model.save_model(out_path)
        print(f"[xgb-cli] saved model to {out_path}")
    print(f"[xgb-cli] trained rows={X.shape[0]} d={X.shape[1]} acc={acc:.3f}")
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse, sys

    ap = argparse.ArgumentParser(description="Train XGBoost on a prompt dataset")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--model-dir", default=".")
    args = ap.parse_args(argv)
    train_xgb_for_prompt(
        args.prompt,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        save_model=not args.no_save,
        model_dir=args.model_dir,
    )
    return 0
