import numpy as np
from latent_state import LatentState


def append_pair(state: LatentState, z_a: np.ndarray, z_b: np.ndarray, label: float) -> None:
    pair = np.stack([z_a.astype(float), z_b.astype(float)], axis=0).reshape(1, 2, state.d)
    zp = getattr(state, 'z_pairs', None)
    ch = getattr(state, 'choices', None)
    state.z_pairs = pair if zp is None else np.vstack([zp, pair])
    state.choices = np.array([label]) if ch is None else np.concatenate([ch, [label]])


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    dfeat = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(dfeat), X.T @ y)

