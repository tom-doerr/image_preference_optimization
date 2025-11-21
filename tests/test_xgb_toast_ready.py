import sys
import types
import numpy as np


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v):
        self[k] = v


def test_xgb_toast_on_sync_fit_via_fit_value_model():
    from constants import Keys
    # Stub streamlit.toast collector
    calls = []
    st = types.ModuleType("streamlit")
    st.toast = lambda msg: calls.append(str(msg))
    sys.modules["streamlit"] = st

    # Stub xgb trainer
    mdl = types.SimpleNamespace(predict_proba=lambda X: np.array([[0.4, 0.6]]))
    xv = types.ModuleType("xgb_value")
    xv.fit_xgb_classifier = lambda X, y, n_estimators=50, max_depth=3: mdl
    sys.modules["xgb_value"] = xv

    from value_model import fit_value_model

    lstate = types.SimpleNamespace(d=4, w=np.zeros(4), w_lock=None)
    X = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=float)
    y = np.array([1, -1], dtype=int)
    ss = SS()
    ss[Keys.XGB_TRAIN_ASYNC] = False  # force sync fit path

    fit_value_model("XGBoost", lstate, X, y, 1.0, ss)
    assert ss.get("xgb_toast_ready") is True


def test_xgb_toast_on_ensure_fitted_sync():
    # Stub streamlit.toast collector
    calls = []
    st = types.ModuleType("streamlit")
    st.toast = lambda msg: calls.append(str(msg))
    sys.modules["streamlit"] = st

    # Stub xgb trainer
    mdl = types.SimpleNamespace(predict_proba=lambda X: np.array([[0.4, 0.6]]))
    xv = types.ModuleType("xgb_value")
    xv.fit_xgb_classifier = lambda X, y, n_estimators=50, max_depth=3: mdl
    sys.modules["xgb_value"] = xv

    from value_model import ensure_fitted

    lstate = types.SimpleNamespace(d=4, w=np.zeros(4), w_lock=None)
    X = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=float)
    y = np.array([1, -1], dtype=int)
    ss = SS()

    ensure_fitted("XGBoost", lstate, X, y, 1.0, ss)
    assert ss.get("xgb_toast_ready") is True
