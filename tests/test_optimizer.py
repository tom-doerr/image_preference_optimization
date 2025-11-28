"""Tests for optimizer classes."""

import numpy as np

from ipo.core.optimizer import (
    HillClimbOptimizer,
    LineSearchOptimizer,
    RidgeOptimizer,
    SPSAOptimizer,
    create_optimizer,
)


def quadratic(z):
    """Simple quadratic: max at z=0."""
    return -np.sum(z ** 2)


def test_spsa_improves():
    z = np.array([1.0, 1.0, 1.0])
    opt = SPSAOptimizer(eta=0.1, eps=0.1, momentum=0.0, verbose=False)
    result = opt.optimize(z, quadratic, n=50)
    assert result.final_value > result.initial_value


def test_spsa_momentum():
    z = np.array([1.0, 1.0, 1.0])
    opt = SPSAOptimizer(eta=0.1, eps=0.1, momentum=0.9, verbose=False)
    result = opt.optimize(z, quadratic, n=50)
    assert result.final_value > result.initial_value


def test_ridge_follows_gradient():
    z = np.array([1.0, 1.0, 1.0])
    w = -2 * z  # gradient of quadratic at z
    opt = RidgeOptimizer(w=w, eta=0.1, verbose=False)
    result = opt.optimize(z, quadratic, n=10)
    assert result.final_value > result.initial_value


def test_line_search():
    z = np.array([1.0, 1.0, 1.0])
    opt = LineSearchOptimizer(direction=-z, eta=0.1, verbose=False)
    result = opt.optimize(z, quadratic, n=10)
    assert result.final_value > result.initial_value


def test_hill_climb():
    z = np.array([1.0, 1.0, 1.0])
    opt = HillClimbOptimizer(sigma=1.0, eta=0.1, verbose=False)
    result = opt.optimize(z, quadratic, n=100)
    assert result.final_value >= result.initial_value


def test_trust_region():
    z = np.array([1.0, 1.0, 1.0])
    opt = SPSAOptimizer(eta=1.0, max_dist=0.5, verbose=False)
    result = opt.optimize(z, quadratic, n=20)
    assert result.distance <= 0.5 + 1e-6


def test_factory():
    assert isinstance(create_optimizer("Grad"), SPSAOptimizer)
    assert isinstance(create_optimizer("Hill"), HillClimbOptimizer)


def test_spsa_high_dim():
    """Test SPSA with high-dimensional input (like latent space)."""
    z = np.random.randn(16384)  # typical latent dim
    opt = SPSAOptimizer(eta=0.1, eps=0.5, momentum=0.9, verbose=False)
    result = opt.optimize(z, quadratic, n=10)
    assert result.steps_taken == 10


def test_spsa_100_steps():
    """100 steps should complete quickly."""
    import time
    z = np.random.randn(16384)
    opt = SPSAOptimizer(eta=1.0, eps=0.5, momentum=0.9, verbose=False)
    t0 = time.time()
    opt.optimize(z, quadratic, n=100)
    assert time.time() - t0 < 5.0


def test_slow_value_fn():
    """Simulates slow XGBoost prediction (~40ms each)."""
    import time
    call_count = [0]
    def slow_fn(z):
        call_count[0] += 1
        time.sleep(0.001)  # 1ms to keep test fast
        return -np.sum(z ** 2)
    z = np.random.randn(100)
    opt = SPSAOptimizer(eta=0.1, momentum=0.9, verbose=False)
    opt.optimize(z, slow_fn, n=10)
    assert call_count[0] == 32  # 3 per step (grad+log) + start + end


def test_nan_value_fn():
    """SPSA should handle NaN gracefully."""
    def nan_fn(z):
        return float('nan')
    z = np.array([1.0, 1.0])
    opt = SPSAOptimizer(verbose=False)
    result = opt.optimize(z, nan_fn, n=5)
    assert result.steps_taken == 0


def test_timeout():
    """SPSA should stop on timeout."""
    import time
    def slow_fn(z):
        time.sleep(0.1)
        return -np.sum(z ** 2)
    z = np.array([1.0, 1.0])
    opt = SPSAOptimizer(max_time=0.25, verbose=False)
    t0 = time.time()
    opt.optimize(z, slow_fn, n=100)
    assert time.time() - t0 < 1.0  # should stop early


def test_param_validation():
    """Params should be clamped to valid ranges."""
    opt = SPSAOptimizer(eta=-1, eps=-1, momentum=5.0, verbose=False)
    assert opt.eta > 0
    assert opt.eps > 0
    assert 0 <= opt.momentum < 1.0
