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
        return np.tile(np.array([[0.1, 0.9]], dtype=float), (X.shape[0], 1))


def test_integration_train_xgb_now_sync_sets_model_and_caption():
    # Stub xgboost
    xgb = types.ModuleType("xgboost")
    xgb.XGBClassifier = _DummyXGB
    sys.modules["xgboost"] = xgb

    # Stub streamlit with writes and a button that triggers the XGB sync fit
    from tests.helpers.st_streamlit import stub_with_writes
    st, writes = stub_with_writes()
    def _btn(label, *a, **k):
        return label == "Train XGBoost now (sync)"
    st.sidebar.button = _btn
    sys.modules["streamlit"] = st

    # Minimal lstate and in-memory dataset (so _get_dataset_for_display returns memory rows)
    class LS:
        d = 3
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=float)
        y = np.array([+1, -1, +1, -1], dtype=float)

    lstate = LS()

    from ipo.ui.ui_sidebar import _handle_train_section
    _handle_train_section(st, lstate, "p", "XGBoost")

    # Verify that a live model was stored and that an XGB scorer is available
    assert getattr(st.session_state, "XGB_MODEL", None) is not None
    from ipo.core.value_scorer import get_value_scorer
    scorer, tag = get_value_scorer("XGBoost", lstate, "p", st.session_state)
    assert callable(scorer) and tag == "XGB"

