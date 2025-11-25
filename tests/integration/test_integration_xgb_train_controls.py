import sys
import types
import numpy as np

from tests.helpers.st_streamlit import stub_with_writes


def _stub_xgb_value():
    m = types.ModuleType("ipo.core.xgb_value")

    class _FakeModel:
        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            s = (X[:, 0] > 0).astype(float) * 0.9
            s = s.reshape(-1, 1)
            return np.hstack([1.0 - s, s])

    def fit_xgb_classifier(X, y, n_estimators=50, max_depth=3):
        return _FakeModel()

    def score_xgb_proba(model, fvec):
        fv = np.asarray(fvec, dtype=float).reshape(1, -1)
        return float(model.predict_proba(fv)[0, 1])

    def get_params(ss):
        return 10, 3

    def set_live_model(ss, model, n_rows):
        ss.XGB_MODEL = model

    def get_live_model(ss):
        return getattr(ss, "XGB_MODEL", None)

    m.fit_xgb_classifier = fit_xgb_classifier
    m.score_xgb_proba = score_xgb_proba
    m.get_params = get_params
    m.set_live_model = set_live_model
    m.get_live_model = get_live_model
    return m


def test_xgb_train_controls_trains_and_scorer_ready(monkeypatch):
    # Stub streamlit
    st, writes = stub_with_writes()
    sys.modules["streamlit"] = st

    # Stub xgb_value to avoid real xgboost dependency
    sys.modules["ipo.core.xgb_value"] = _stub_xgb_value()

    # Minimal lstate and dataset
    lstate = types.SimpleNamespace(d=8)
    X = np.vstack([np.eye(8)[:4], -np.eye(8)[:4]])
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)

    # Call the train controls directly
    from ipo.ui.sidebar.panels import _xgb_train_controls
    _xgb_train_controls(st, lstate, X, y)

    # Value scorer should now be available via XGB
    from ipo.core.value_scorer import get_value_scorer
    scorer, status = get_value_scorer("XGBoost", lstate, "prompt", st.session_state)
    assert scorer is not None
    # Predict on a simple feature diff
    v = float(scorer(np.eye(8)[0]))
    assert v > 0.5

