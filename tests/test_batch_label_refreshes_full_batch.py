import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_basic


class TestBatchLabelRefreshesFullBatch(unittest.TestCase):
    def test_label_click_refreshes_entire_batch(self):
        st = stub_basic()
        # Enable batch curation and set batch size 3
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'Batch curation mode' in label else value
            @staticmethod
            def slider(label, *a, **k):
                if 'Batch size' in label:
                    return 3
                return k.get('value', 1)
        st.sidebar = SB()
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        sys.modules['flux_local'] = fl

        # First import: build initial batch
        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        before = [np.copy(z) for z in app.st.session_state.cur_batch]

        # Now simulate clicking Good on item 0
        def _btn(label, *a, **k):
            return label == 'Good (+1) 0'
        st.button = _btn
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app2  # noqa: F401
        after = app.st.session_state.cur_batch

        # Entire batch should be refreshed, not just index 0
        changed = sum(float(np.linalg.norm(b - a)) > 1e-12 for b, a in zip(before, after))
        self.assertGreaterEqual(changed, 2)


if __name__ == '__main__':
    unittest.main()

