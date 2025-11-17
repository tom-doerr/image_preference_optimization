import colorsys
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class State:
    w: np.ndarray
    rng: np.random.Generator
    step: int = 0


def init_state(dim: int = 3, seed: Optional[int] = 0) -> State:
    return State(w=np.zeros(dim, dtype=float), rng=np.random.default_rng(seed))


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def propose_pair(state: State) -> Tuple[np.ndarray, np.ndarray]:
    if state.step == 0:
        a = state.rng.random(3)
        b = state.rng.random(3)
    else:
        base = _sigmoid(state.w)
        noise = state.rng.normal(0.0, 0.15, 3)
        a = _clip01(base + noise)
        b = _clip01(1.0 - base + noise)
    return a, b


def update(state: State, a: np.ndarray, b: np.ndarray, choice: str, lr: float = 0.5) -> None:
    diff = a - b
    if choice == 'a':
        state.w += lr * diff
    elif choice == 'b':
        state.w -= lr * diff
    else:
        raise ValueError("choice must be 'a' or 'b'")
    state.step += 1


def feature_to_image(z: np.ndarray, size: int = 160) -> np.ndarray:
    h, s, v = float(z[0]), float(z[1]), float(z[2])
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[..., 0] = int(r * 255)
    img[..., 1] = int(g * 255)
    img[..., 2] = int(b * 255)
    return img


def estimate_preferred_feature(state: State) -> np.ndarray:
    return _clip01(_sigmoid(state.w))
