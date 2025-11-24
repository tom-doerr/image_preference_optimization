import numpy as np
from typing import Dict


def ridge_cv_accuracy(
    X: np.ndarray, y: np.ndarray, lam: float = 1e-3, k: int = 5, max_rows: int = 64
) -> float:
    """Compute a minimal K-fold CV accuracy for ridge sign classifier.

    - Uses dual ridge (w = X^T (XX^T + λI)^{-1} y) to avoid d×d solves.
    - Caps rows to `max_rows` for speed; uses first rows deterministically.
    - Returns accuracy in [0,1]. Raises on invalid inputs.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = int(X.shape[0])
    if n == 0:
        return float("nan")
    if n > max_rows:
        X = X[:max_rows]
        y = y[:max_rows]
        n = max_rows
    k = max(2, min(int(k), n))
    fold_sizes = [(n // k) + (1 if i < (n % k) else 0) for i in range(k)]
    idx = np.arange(n)
    pos = 0
    correct = 0
    for fs in fold_sizes:
        val_idx = idx[pos : pos + fs]
        tr_idx = (
            np.concatenate([idx[:pos], idx[pos + fs :]])
            if fs < n
            else np.array([], dtype=int)
        )
        pos += fs
        if tr_idx.size == 0 or val_idx.size == 0:
            continue
        Xtr, ytr = X[tr_idx], y[tr_idx]
        K = Xtr @ Xtr.T
        K.ravel()[:: K.shape[1] + 1] += float(lam)
        alpha = np.linalg.solve(K, ytr)
        w = Xtr.T @ alpha
        y_pred = np.sign(X[val_idx] @ w)
        correct += int((y_pred == np.sign(y[val_idx])).sum())
    return correct / float(n)


def pair_metrics(
    w: np.ndarray, z_a: np.ndarray, z_b: np.ndarray
) -> Dict[str, float | str]:
    za_n = float(np.linalg.norm(z_a))
    zb_n = float(np.linalg.norm(z_b))
    diff = z_b - z_a
    diff_n = float(np.linalg.norm(diff))
    w_n = float(np.linalg.norm(w))
    if w_n > 0.0 and diff_n > 0.0:
        c = float(np.dot(w, diff) / (w_n * diff_n))
    else:
        c = float("nan")
    return {
        "za_norm": za_n,
        "zb_norm": zb_n,
        "diff_norm": diff_n,
        "cos_w_diff": c,
    }


def xgb_cv_accuracy(
    X: np.ndarray, y: np.ndarray, k: int = 3, n_estimators: int = 50, max_depth: int = 3
) -> float:
    """Tiny XGBoost-based K-fold CV accuracy.

    - Deterministic shuffle + split into k folds (k clamped to [2, n]).
    - Trains a fresh XGB model per fold via xgb_value.fit_xgb_classifier.
    - Uses xgb_value.score_xgb_proba to score the held-out fold and computes
      accuracy vs labels (y>0).
    - Returns mean accuracy over non-empty folds in [0,1]; NaN if no folds.
    """
    import numpy as _np
    from ipo.core.xgb_value import fit_xgb_classifier, score_xgb_proba  # type: ignore

    X = _np.asarray(X, dtype=float)
    y = _np.asarray(y, dtype=float).ravel()
    n = int(X.shape[0])
    if n == 0:
        return float("nan")
    k = max(2, min(int(k), n))
    idx = _np.arange(n)
    rng = _np.random.default_rng(0)
    rng.shuffle(idx)
    folds = _np.array_split(idx, k)
    accs: list[float] = []
    for fi in range(k):
        test_idx = folds[fi]
        train_idx = _np.concatenate([folds[j] for j in range(k) if j != fi])
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        mdl = fit_xgb_classifier(
            X[train_idx], y[train_idx], n_estimators=n_estimators, max_depth=max_depth
        )
        probs = _np.array([score_xgb_proba(mdl, fv) for fv in X[test_idx]], dtype=float)
        preds = probs >= 0.5
        accs.append(float(_np.mean(preds == (y[test_idx] > 0))))
    if not accs:
        return float("nan")
    return float(_np.mean(accs))
