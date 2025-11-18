import numpy as np
from typing import Dict


def ridge_cv_accuracy(X: np.ndarray, y: np.ndarray, lam: float = 1e-3, k: int = 5, max_rows: int = 64) -> float:
    """Compute a minimal K-fold CV accuracy for ridge sign classifier.

    - Uses dual ridge (w = X^T (XX^T + λI)^{-1} y) to avoid d×d solves.
    - Caps rows to `max_rows` for speed; uses first rows deterministically.
    - Returns accuracy in [0,1]. Raises on invalid inputs.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = int(X.shape[0])
    if n == 0:
        return float('nan')
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
        val_idx = idx[pos:pos + fs]
        tr_idx = np.concatenate([idx[:pos], idx[pos + fs:]]) if fs < n else np.array([], dtype=int)
        pos += fs
        if tr_idx.size == 0 or val_idx.size == 0:
            continue
        Xtr, ytr = X[tr_idx], y[tr_idx]
        K = Xtr @ Xtr.T
        K.ravel()[::K.shape[1]+1] += float(lam)
        alpha = np.linalg.solve(K, ytr)
        w = Xtr.T @ alpha
        y_pred = np.sign(X[val_idx] @ w)
        correct += int((y_pred == np.sign(y[val_idx])).sum())
    return correct / float(n)


def pair_metrics(w: np.ndarray, z_a: np.ndarray, z_b: np.ndarray) -> Dict[str, float | str]:
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
