"""Sampling strategies for latent space exploration."""
import numpy as np
from typing import Optional, Tuple


def get_good_mean(X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """Mean of positively-labeled samples."""
    if X is not None and y is not None and (y > 0).sum() > 0:
        return X[y > 0].mean(axis=0)
    return None


def get_good_dist(X: np.ndarray, y: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Per-dim mean/std from good samples."""
    if X is not None and y is not None and (y > 0).sum() >= 2:
        Xg = X[y > 0]
        return Xg.mean(axis=0), Xg.std(axis=0) + 1e-6
    return None, None


def random_offset(d: int, sigma: float, rng: np.random.Generator, scale: float = 0.8) -> np.ndarray:
    """Random unit direction scaled by sigma."""
    r = rng.standard_normal(d)
    return sigma * scale * r / (np.linalg.norm(r) + 1e-12)
