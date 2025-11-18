import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestRegLambda(unittest.TestCase):
    def test_ridge_lambda_passed_to_trainer(self):
        st = stub_basic(pre_images=False)
        # Enable curation and set batch size + ridge λ
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'Batch curation mode' in label else value
            @staticmethod
            def slider(label, *a, **k):
                if 'Batch size' in label:
                    return 2
                if 'Ridge' in label:
                    return 0.05
                return k.get('value', 1)
        st.sidebar = SB()
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **kw: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        import app
        # Add two labels so trainer runs
        zs = app.st.session_state.cur_batch
        app._curation_add(1, zs[0])
        app._curation_add(-1, zs[1])

        # Patch ridge_fit to capture lam
        import latent_logic as ll
        called = {}
        def _rf(X, y, lam):
            called['lam'] = lam
            # return zeros of correct shape
            import numpy as np
            return np.zeros(X.shape[1], dtype=float)
        ll.ridge_fit = _rf

        app._curation_train_and_next()
        assert abs(called.get('lam', 0) - 0.05) < 1e-9


if __name__ == '__main__':
    unittest.main()

