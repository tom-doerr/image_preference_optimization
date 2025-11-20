import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_with_writes


class _XGBCls:
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        # simple heuristic: positive on sum>0
        return np.stack(
            [
                1.0 - (X.sum(axis=1) > 0).astype(float),
                (X.sum(axis=1) > 0).astype(float),
            ],
            axis=1,
        )


class TestXGBTrainScoreSidebar(unittest.TestCase):
    def test_train_score_shown_for_xgb(self):
        st, writes = stub_with_writes()

        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, options, index=0):
                if "Value model" in label:
                    return "XGBoost"
                if "Generation mode" in label:
                    return "Batch curation"
                return options[index]

        st.sidebar = SB()
        sys.modules["streamlit"] = st

        # Stubs
        xgb = types.SimpleNamespace(XGBClassifier=_XGBCls)
        sys.modules["xgboost"] = xgb
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **kw: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        # First import
        if "app" in sys.modules:
            del sys.modules["app"]
        import app

        # Inject small separable dataset
        X = np.vstack([np.eye(8)[:4], -np.eye(8)[:4]])
        y = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
        app.st.session_state.lstate.X = X
        app.st.session_state.lstate.y = y
        # Re-render to train XGB and show score
        if "app" in sys.modules:
            del sys.modules["app"]
        import app as app2  # noqa: F401

        out = "\n".join(writes)
        self.assertIn("Value model: XGBoost", out)
        # Re-import again to ensure score reflects XGB path
        if "app" in sys.modules:
            del sys.modules["app"]
        import app as app3  # noqa: F401

        out = "\n".join(writes)
        self.assertIn("Train score:", out)


if __name__ == "__main__":
    unittest.main()
