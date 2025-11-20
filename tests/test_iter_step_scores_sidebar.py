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
        prompt = 'unit test prompt'
        st.session_state['prompt'] = prompt

        # Stub flux_local
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.generate_flux_image = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        sys.modules['streamlit'] = st
        # Ensure no stale on-disk dataset for this prompt
        from persistence import dataset_path_for_prompt
        path = dataset_path_for_prompt(prompt)
        import os
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401
        # Persist a tiny dataset for this prompt so helper can load it
        from persistence import append_dataset_row
        d = app.st.session_state.lstate.d
        feat_pos = np.zeros((1, d), dtype=float); feat_pos[0, 0] = 1.0
        feat_neg = np.zeros((1, d), dtype=float); feat_neg[0, 1] = -1.0
        append_dataset_row(st.session_state['prompt'], feat_pos, +1.0)
        append_dataset_row(st.session_state['prompt'], feat_neg, -1.0)
        # Ensure w is non-zero, then render scores now that state exists
        app.st.session_state.lstate.w = np.ones(app.st.session_state.lstate.d, dtype=float)
        app._render_iter_step_scores()
        out = "\n".join(writes)
        self.assertTrue(any('Step scores:' in w or 'Step 1' in w for w in writes), out)


if __name__ == '__main__':
    unittest.main()
