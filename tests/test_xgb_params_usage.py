import io
import sys
import types
import numpy as np


def _capture(func, *a, **k):
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        func(*a, **k)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_ensure_fitted_uses_xgb_params_from_session():
from value_model import ensure_fitted

    # Stub xgb trainer and capture params implicitly via debug line
    xv = types.ModuleType("xgb_value")
    xv.fit_xgb_classifier = lambda X, y, n_estimators=50, max_depth=3: object()
    sys.modules["xgb_value"] = xv

    lstate = types.SimpleNamespace(d=4, w=np.zeros(4))
    X = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=float)
    y = np.array([1, -1], dtype=int)

    class SS(dict):
        __getattr__ = dict.get
    ss = SS()
    ss["xgb_n_estimators"] = 77
    ss["xgb_max_depth"] = 5

    out = _capture(ensure_fitted, "XGBoost", lstate, X, y, 1.0, ss)
    assert "n_estim=77" in out and "depth=5" in out

