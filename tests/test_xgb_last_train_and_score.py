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
        # positive when sum>0 → makes dataset perfectly separable
        pos = (X.sum(axis=1) > 0).astype(float)
        return np.stack([1.0 - pos, pos], axis=1)


class TestXGBLastTrainAndScore(unittest.TestCase):
    def _setup_stubs(self):
        st, writes = stub_with_writes()
        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, options, index=0):
                if 'Value model' in label:
                    return 'XGBoost'
                if 'Generation mode' in label:
                    return 'Batch curation'
                return options[index]
        st.sidebar = SB()
        # capture writes/metrics
        st.sidebar.write = lambda *a, **k: writes.append(str(a[0]) if a else "")
        st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
        sys.modules['streamlit'] = st

        # xgboost + flux stubs
        xgb = types.SimpleNamespace(XGBClassifier=_XGBCls)
        sys.modules['xgboost'] = xgb
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **kw: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl
        return st, writes

    def test_xgb_last_train_timestamp_and_score(self):
        st, writes = self._setup_stubs()
        # First import with XGB selected
        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        # Inject a tiny separable dataset
        X = np.vstack([np.eye(8)[:4], -np.eye(8)[:4]])
        y = np.array([1,1,1,1,-1,-1,-1,-1], dtype=float)
        app.st.session_state.lstate.X = X
        app.st.session_state.lstate.y = y
        # Re-render to fit XGB
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app2  # noqa: F401
        # Render once more to ensure Data block reflects fit
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app3  # noqa: F401

        out = "\n".join(writes)
        # Value model line present and set to XGBoost
        self.assertIn('Value model: XGBoost', out)
        # Train score appears
        self.assertIn('Train score:', out)
        # Expect perfect accuracy for our separable dataset
        self.assertIn('Train score: 100%', out)
        # Last train shows a timestamp (not n/a)
        last_lines = [ln for ln in out.splitlines() if ln.startswith('Last train:')]
        self.assertTrue(last_lines and last_lines[-1] != 'Last train: n/a')


if __name__ == '__main__':
    unittest.main()

