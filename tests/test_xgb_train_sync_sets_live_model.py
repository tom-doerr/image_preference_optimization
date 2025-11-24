import sys
import types
import numpy as np


class _DummyXGB:
    def __init__(self, **k):
        self.k = k

    def fit(self, X, y):
        self.n = int(getattr(X, "shape", (0,))[0])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.tile(np.array([[0.2, 0.8]], dtype=float), (X.shape[0], 1))


def _stub_xgboost_module():
    mod = types.ModuleType("xgboost")
    mod.XGBClassifier = _DummyXGB
    sys.modules["xgboost"] = mod


def test_xgb_train_controls_sets_live_model_and_scorer_ok():
    _stub_xgboost_module()
    from tests.helpers.st_streamlit import stub_with_writes
    st, _ = stub_with_writes()

    # Minimal lstate and dataset with both classes
    class LS:
        d = 3

    lstate = LS()
    X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=float)
    y = np.array([+1, -1, +1, -1], dtype=float)

    from ipo.ui.ui_sidebar import _xgb_train_controls

    _xgb_train_controls(st, lstate, X, y)

    # Live model should be available now
    assert getattr(st.session_state, "XGB_MODEL", None) is not None

    from ipo.core.value_scorer import get_value_scorer
    scorer, tag = get_value_scorer("XGBoost", lstate, "p", st.session_state)
    assert callable(scorer)
    assert tag == "XGB"

