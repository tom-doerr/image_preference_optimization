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
            @staticmethod
            def selectbox(label, options, index=0):
                # Force Ridge to avoid XGB import/training in the unit test
                if 'Value model' in label:
                    return 'Ridge'
                return options[index]
        st.sidebar = SB()
        sys.modules['streamlit'] = st
        # Fast stubs
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **k: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        # Folder dataset is new per unique prompt; no NPZ cleanup needed
        import app
        zs = app.st.session_state.cur_batch
        # Append two labeled items → saved to file by helper
        app._curation_add(1, zs[0])
        app._curation_add(-1, zs[1])
        # Capture that ridge_fit is called with shapes matching file contents
        # Patch latent_logic.ridge_fit (used by value_model.fit_value_model)
        called = {}
        import latent_logic as ll
        _orig_rf = ll.ridge_fit
        def _rf(X, y, lam):
            called['n'] = X.shape[0]
            return _orig_rf(X, y, lam)
        ll.ridge_fit = _rf
        app._curation_train_and_next()
        self.assertTrue(called.get('n', 0) >= 2)
        # restore
        ll.ridge_fit = _orig_rf


if __name__ == '__main__':
    unittest.main()
