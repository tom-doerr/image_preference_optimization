import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_with_writes


class TestStepScoresRidgeNonZero(unittest.TestCase):
    def test_ridge_weights_yield_nonzero_scores(self):
        st, writes = stub_with_writes()
        st.session_state.prompt = 'step-scores-ridge'
        st.sidebar.selectbox = staticmethod(lambda *a, **k: 'Ridge')
        sys.modules['streamlit'] = st
        # Minimal app deps
        fl = types.ModuleType('flux_local')
        fl.set_model = lambda *a, **k: None
        fl.generate_flux_image_latents = lambda *a, **k: 'img'
        sys.modules['flux_local'] = fl
        import app as appmod
        # Set a simple nonzero weight vector, then clear prior writes
        d = appmod.st.session_state.lstate.d
        appmod.st.session_state.lstate.w = np.ones(d, dtype=float)
        writes.clear()
        # Re-render sidebar step scores
        del sys.modules['app']
        sys.modules['streamlit'] = st
        sys.modules['flux_local'] = fl
        import app  # noqa: F401
        out = "\n".join(writes)
        self.assertIn('Step scores:', out)
        self.assertNotIn('Step scores: n/a', out)
        # Expect non-zero values (not all 0.000)
        self.assertNotIn('Step scores: 0.000', out)


if __name__ == '__main__':
    unittest.main()
