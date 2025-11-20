import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class _XGBCls:
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        import numpy as np

        return np.tile([0.4, 0.6], (X.shape[0], 1))


class TestE2EValueModelDropdown(unittest.TestCase):
    def test_dropdown_switches_to_xgb(self):
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
        # preserve writes capture
        st.sidebar.write = lambda *a, **k: writes.append(str(a[0]) if a else "")
        st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
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

        if "app" in sys.modules:
            del sys.modules["app"]
        import app

        out = "\n".join(writes)
        self.assertIn("Value model: XGBoost", out)
        self.assertTrue(getattr(app, "use_xgb", False))


if __name__ == "__main__":
    unittest.main()
