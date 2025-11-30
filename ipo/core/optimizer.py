"""Latent space optimizers for value function maximization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class OptimResult:
    """Result of optimization."""
    z: np.ndarray
    initial_value: float
    final_value: float
    steps_taken: int
    distance: float


class Optimizer(ABC):
    """Base class for latent space optimizers."""

    def __init__(self, max_dist: float = 0.0, max_time: float = 30.0, verbose: bool = True):
        self.max_dist = max_dist
        self.max_time = max_time
        self.verbose = verbose

    @abstractmethod
    def optimize(self, z: np.ndarray, value_fn: Callable, n: int) -> OptimResult:
        pass

    def _clip(self, z: np.ndarray, z0: np.ndarray) -> np.ndarray:
        if self.max_dist <= 0:
            return z
        d = np.linalg.norm(z - z0)
        return z0 + (z - z0) * self.max_dist / d if d > self.max_dist else z

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


class SPSAOptimizer(Optimizer):
    """SPSA gradient approximation optimizer."""

    def __init__(  # noqa: E501
        self, eta=1.0, eps=0.5, momentum=0.9, max_dist=0.0, max_time=30.0, seed=None, verbose=True
    ):
        super().__init__(max_dist, max_time, verbose)
        self.eta = max(1e-6, float(eta))
        self.eps = max(1e-6, float(eps))
        self.momentum = max(0.0, min(0.99, float(momentum)))
        self.rng = np.random.default_rng(seed)

    def optimize(self, z: np.ndarray, value_fn: Callable, n: int) -> OptimResult:
        import time
        t0 = time.time()
        z0, best, vel = z.copy(), z.copy(), np.zeros_like(z)
        v0 = value_fn(z)
        if not np.isfinite(v0):
            self._log("[spsa] initial value not finite, aborting")
            return OptimResult(z, 0.0, 0.0, 0, 0.0)
        self._log(f"[spsa] start v={v0:.4f} eps={self.eps} eta={self.eta}")
        for i in range(n):
            if time.time() - t0 > self.max_time:
                self._log(f"[spsa] timeout at step {i}")
                break
            delta = self.rng.standard_normal(len(best))
            delta = delta / (np.linalg.norm(delta) + 1e-12)
            fp = value_fn(best + self.eps * delta)
            fm = value_fn(best - self.eps * delta)
            grad = (fp - fm) / (2 * self.eps) * delta
            vel = self.momentum * vel + grad
            best = self._clip(best + self.eta * vel, z0)
            v = value_fn(best)
            self._log(f"[spsa] step {i}: v={v:.4f}")
            if not np.all(np.isfinite(best)):
                self._log(f"[spsa] NaN/Inf at step {i}, stopping")
                best = z0
                break
        vf = value_fn(best)
        dist = np.linalg.norm(best - z0)
        self._log(f"[spsa] done {v0:.4f}->{vf:.4f} d={dist:.1f}")
        return OptimResult(best, v0, vf, n, dist)


class RidgeOptimizer(Optimizer):
    """Gradient ascent using Ridge weights."""

    def __init__(self, w: np.ndarray, eta=0.01, max_dist=0.0, max_time=30.0, verbose=True):
        super().__init__(max_dist, max_time, verbose)
        self.w = np.asarray(w, dtype=float)
        self.eta = max(1e-6, float(eta))

    def optimize(self, z: np.ndarray, value_fn: Callable, n: int) -> OptimResult:
        z0, best, v0 = z.copy(), z.copy(), value_fn(z)
        wn = np.linalg.norm(self.w)
        if wn < 1e-12:
            return OptimResult(z, v0, v0, 0, 0.0)
        w_dir = self.w / wn
        for i in range(n):
            best = self._clip(best + self.eta * w_dir, z0)
            v = value_fn(best)
            self._log(f"[ridge] step {i}: v={v:.4f}")
        vf = value_fn(best)
        dist = np.linalg.norm(best - z0)
        self._log(f"[ridge] {v0:.4f}->{vf:.4f} d={dist:.1f}")
        return OptimResult(best, v0, vf, n, dist)


class LineSearchOptimizer(Optimizer):
    """Line search along a direction."""

    def __init__(self, direction: np.ndarray, eta=0.1, max_dist=0.0, max_time=30.0, verbose=True):
        super().__init__(max_dist, max_time, verbose)
        direction = np.asarray(direction, dtype=float)
        dn = np.linalg.norm(direction)
        self.direction = direction / (dn + 1e-12)
        self.eta = max(1e-6, float(eta))

    def optimize(self, z: np.ndarray, value_fn: Callable, n: int) -> OptimResult:
        z0, best = z.copy(), z.copy()
        v0 = best_val = value_fn(z)
        for i in range(n):
            cand = self._clip(z + self.eta * (i + 1) * self.direction, z0)
            v = value_fn(cand)
            self._log(f"[line] step {i}: v={v:.4f}")
            if v > best_val:
                best, best_val = cand, v
        dist = np.linalg.norm(best - z0)
        self._log(f"[line] {v0:.4f}->{best_val:.4f} d={dist:.1f}")
        return OptimResult(best, v0, best_val, n, dist)


class HillClimbOptimizer(Optimizer):
    """Random hill climbing."""

    def __init__(self, sigma=1.0, eta=0.1, max_dist=0.0, max_time=30.0, seed=None, verbose=True):
        super().__init__(max_dist, max_time, verbose)
        self.sigma = max(1e-6, float(sigma))
        self.eta = max(1e-6, float(eta))
        self.rng = np.random.default_rng(seed)

    def optimize(self, z: np.ndarray, value_fn: Callable, n: int) -> OptimResult:
        z0, best, best_val = z.copy(), z.copy(), value_fn(z)
        v0 = best_val
        for i in range(n):
            cand = best + self.rng.standard_normal(len(best)) * self.eta * self.sigma
            cand = self._clip(cand, z0)
            v = value_fn(cand)
            self._log(f"[hill] step {i}: v={v:.4f}")
            if v > best_val:
                best, best_val = cand, v
        dist = np.linalg.norm(best - z0)
        self._log(f"[hill] {v0:.4f}->{best_val:.4f} d={dist:.1f}")
        return OptimResult(best, v0, best_val, n, dist)


def create_optimizer(mode: str, **kw) -> Optimizer:
    """Factory to create optimizer by mode name."""
    opts = {"Grad": SPSAOptimizer, "Ridge": RidgeOptimizer,
            "Line": LineSearchOptimizer, "Hill": HillClimbOptimizer}
    if mode not in opts:
        raise ValueError(f"Unknown mode: {mode}")
    return opts[mode](**kw)
