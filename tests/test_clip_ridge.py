"""Tests for CLIP mode ridge regression."""
import numpy as np
import pytest


def test_ridge_fit_basic():
    """Ridge fit returns weights."""
    from ipo.ui.clip_mode import _ridge_fit
    X = np.random.randn(10, 5)
    y = np.random.randn(10)
    w = _ridge_fit(X, y, alpha=1.0)
    assert w.shape == (5,)


def test_ridge_fit_small_data():
    """Ridge fit returns None for <2 samples."""
    from ipo.ui.clip_mode import _ridge_fit
    X = np.random.randn(1, 5)
    y = np.random.randn(1)
    assert _ridge_fit(X, y) is None


def test_ridge_cv_returns_tuple():
    """Ridge CV returns (weights, alpha, scores)."""
    from ipo.ui.clip_mode import _ridge_cv
    X = np.random.randn(20, 5)
    y = np.random.randn(20)
    w, alpha, scores = _ridge_cv(X, y)
    assert w.shape == (5,)
    assert alpha > 0
    assert isinstance(scores, dict)


def test_alphas_constant():
    """ALPHAS constant has expected values."""
    from ipo.ui.clip_mode import ALPHAS
    assert 0.01 in ALPHAS
    assert 1.0 in ALPHAS
    assert 1000.0 in ALPHAS


def test_ridge_cv_small_data_fallback():
    """Ridge CV falls back for <4 samples."""
    from ipo.ui.clip_mode import _ridge_cv
    X = np.random.randn(3, 5)
    y = np.random.randn(3)
    w, alpha, scores = _ridge_cv(X, y)
    assert w.shape == (5,)
    assert alpha == 1.0
    assert scores == {}
