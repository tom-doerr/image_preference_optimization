import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestTrainFromSavedDataset(unittest.TestCase):
    def test_trains_from_file_after_adds(self):
        st = stub_basic(pre_images=False)
        st.text_input = lambda *_, value="": 'train ds test'
        # Enable curation mode to access helpers
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'Batch curation mode' in label else value
            @staticmethod
            def slider(label, *a, **k):
                if 'Batch size' in label:
                    return 2
                if 'Ridge' in label:
                    return 0.01
                return k.get('value', 1)
        st.sidebar = SB()
        sys.modules['streamlit'] = st
        # Fast stubs
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **k: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        import os
        from persistence import dataset_path_for_prompt
        # Ensure clean dataset file for this prompt
        p = 'train ds test'
        try:
            os.remove(dataset_path_for_prompt(p))
        except Exception:
            pass
        import app
        zs = app.st.session_state.cur_batch
        # Append two labeled items → saved to file by helper
        app._curation_add(1, zs[0])
        app._curation_add(-1, zs[1])
        # Capture that ridge_fit is called with shapes matching file contents
        import app as appmod
        called = {}
        def _rf(X, y, lam):
            called['n'] = X.shape[0]
            return X[0] * 0.0
        appmod.ridge_fit = _rf
        app._curation_train_and_next()
        self.assertEqual(called.get('n'), 2)


if __name__ == '__main__':
    unittest.main()
