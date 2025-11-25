import sys
import types
import numpy as np

from tests.helpers.st_streamlit import stub_click_button


def _stub_pipeline_local():
    m = types.ModuleType("ipo.infra.pipeline_local")
    m.set_model = lambda *a, **k: None
    m.get_last_call = lambda: {}
    m.generate_flux_image_latents = lambda *a, **k: "ok-image"
    return m


def _stub_xgb_value():
    m = types.ModuleType("ipo.core.xgb_value")

    class _FakeModel:
        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            # Simple linear score on the first feature
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


def test_integration_xgb_sync_button_trains_and_enables_captions(monkeypatch):
    # Streamlit stub that clicks the XGB sync button and captures image captions
    st = stub_click_button("Train XGBoost now (sync)")

    captions = []

    def _image(*a, **k):
        cap = k.get("caption") or ""
        captions.append(str(cap))

    st.image = _image

    class SB(st.sidebar.__class__):
        @staticmethod
        def selectbox(label, options, index=0):
            if "Value model" in label:
                return "XGBoost"
            if "Generation mode" in label:
                return "Batch curation"
            return options[index]

    st.sidebar = SB()
    st.sidebar.button = staticmethod(lambda label, *a, **k: label == "Train XGBoost now (sync)")
    st.sidebar_writes = []
    st.sidebar.write = lambda *a, **k: st.sidebar_writes.append(str(a[0]) if a else "")
    sys.modules["streamlit"] = st

    # Stub heavy modules
    sys.modules["ipo.infra.pipeline_local"] = _stub_pipeline_local()
    sys.modules["ipo.core.xgb_value"] = _stub_xgb_value()

    # First import to build lstate
    if "app" in sys.modules:
        del sys.modules["app"]
    import app  # noqa: F401

    # Populate an in-memory dataset with both classes
    ls = st.session_state.lstate
    X = np.vstack([np.eye(8)[:4], -np.eye(8)[:4]])
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    ls.X = X
    ls.y = y

    # Re-import to trigger sidebar and the clicked Train XGBoost button
    del sys.modules["app"]
    import app as app2  # noqa: F401

    # Assert model is set and captions include [XGB]
    mdl = getattr(st.session_state, "XGB_MODEL", None)
    assert mdl is not None, "XGB model not stored in session"
    # At least one tile caption should contain [XGB]
    assert any("[XGB]" in c for c in captions) or any("Value model: XGBoost" in w for w in st.sidebar_writes)
