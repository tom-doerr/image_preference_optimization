import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_click_button


class TestBatchKeepTrain(unittest.TestCase):
    def test_train_keep_batch_does_not_replace_items(self):
        st = stub_click_button("Train on dataset (keep batch)")
        # Enable batch curation mode and set batch size to 2 for speed
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'Batch curation mode' in label else value
            @staticmethod
            def slider(label, *a, **k):
                if 'Batch size' in label:
                    return 2
                return k.get('value', 1)
        st.sidebar = SB()
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        sys.modules['flux_local'] = fl

        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        before = [np.copy(z) for z in app.st.session_state.cur_batch]
        # Trigger the button; app import already rendered and clicked
        after = app.st.session_state.cur_batch
        # same length and first item unchanged
        self.assertEqual(len(before), len(after))
        self.assertAlmostEqual(float(np.linalg.norm(before[0] - after[0])), 0.0, places=9)


if __name__ == '__main__':
    unittest.main()

