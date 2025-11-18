import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarCVScore(unittest.TestCase):
    def test_cv_score_shown_when_inmemory_data_available(self):
        st, writes = stub_with_writes()
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        # Import app first, then inject small in-memory dataset and re-render Data block via simple write calls
        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        # Build tiny linearly-separable dataset in low dimension (8)
        X = np.vstack([np.eye(8)[:4], -np.eye(8)[:4]])
        y = np.array([1,1,1,1,-1,-1,-1,-1], dtype=float)
        app.st.session_state.lstate.X = X
        app.st.session_state.lstate.y = y
        # Force a second import to re-run sidebar (simple approach under our stubs)
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app2  # noqa: F401

        out = "\n".join(writes)
        self.assertIn("CV score", out)


if __name__ == '__main__':
    unittest.main()

