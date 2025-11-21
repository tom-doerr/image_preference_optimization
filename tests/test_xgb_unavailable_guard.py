import sys
import types
import numpy as np


def test_xgb_unavailable_sets_status_and_skips_resubmit():
    from value_model import ensure_fitted, fit_value_model
    from constants import Keys

    # Stub trainer to raise ImportError
    xv = types.ModuleType("xgb_value")
    def _raise(*a, **k):
        raise ImportError("xgboost missing")
    xv.fit_xgb_classifier = _raise
    sys.modules["xgb_value"] = xv

    lstate = types.SimpleNamespace(d=4, w=np.zeros(4))
    X = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=float)
    y = np.array([1, -1], dtype=int)
    ss = {}

    ensure_fitted("XGBoost", lstate, X, y, 1.0, ss)
    st = ss.get(Keys.XGB_TRAIN_STATUS)
    assert isinstance(st, dict) and st.get("state") == "xgb_unavailable"

    # Second call should skip resubmit
    before = ss.copy()
    fit_value_model("XGBoost", lstate, X, y, 1.0, ss)
    after = ss.copy()
    assert after.get(Keys.XGB_TRAIN_STATUS, {}).get("state") == "xgb_unavailable"

