import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_with_writes


class TestIterStepScoresSidebar(unittest.TestCase):
    def test_step_scores_render_for_distancehill(self):
        st, writes = stub_with_writes()

        st.sidebar.selectbox = staticmethod(lambda label, options, index=0: (
            'Batch curation' if 'Generation mode' in label else (
            'DistanceHill' if 'Value model' in label else (
            options[0] if 'Model' in label else options[index]
        ))) )
        st.session_state['prompt'] = 'unit test prompt'

        # Provide a tiny dataset and non-zero w so scores are computed
        # Create a 512x512 default state later; d = 16384
        d = 16384
        X = np.zeros((2, d), dtype=float)
        X[0, 0] = 1.0
        X[1, 1] = -1.0
        y = np.array([1.0, -1.0], dtype=float)
        st.session_state['dataset_X'] = X
        st.session_state['dataset_y'] = y

        # Stub flux_local
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.generate_flux_image = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        sys.modules['streamlit'] = st
        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401
        # Ensure w is non-zero, then render scores now that state exists
        app.st.session_state.lstate.w = np.ones(app.st.session_state.lstate.d, dtype=float)
        app._render_iter_step_scores()
        out = "\n".join(writes)
        self.assertTrue(any('Step scores:' in w or 'Step 1' in w for w in writes), out)


if __name__ == '__main__':
    unittest.main()
