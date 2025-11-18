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


class TestXGBOption(unittest.TestCase):
    def test_checkbox_enables_xgb_without_crash(self):
        st, writes = stub_with_writes()
        # Enable XGBoost option
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'XGBoost' in label else value
        st.sidebar = SB()
        sys.modules['streamlit'] = st
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **kw: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl
        # Stub xgboost module
        xgb = types.SimpleNamespace(XGBClassifier=_XGBCls)
        sys.modules['xgboost'] = xgb

        import app
        # Ensure app imported and option toggled
        self.assertTrue(getattr(app, 'use_xgb', False))


if __name__ == '__main__':
    unittest.main()
