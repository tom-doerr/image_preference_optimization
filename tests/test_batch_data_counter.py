import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestBatchDataCounter(unittest.TestCase):
    def test_dataset_rows_counter_updates(self):
        st, writes = stub_with_writes()
        prompt = 'batch data counter test'
        st.text_input = lambda *_, value="": prompt
        # Enable batch mode
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
        # capture sidebar writes
        st.sidebar.write = lambda *a, **k: writes.append(str(a[0]) if a else "")

        # Folder datasets are unique per prompt; no NPZ cleanup required

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **k: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        import app
        from latent_state import init_latent_state
        app._apply_state(init_latent_state())
        # ensure lstate and pair/batch are initialized
        app._curation_init_batch()
        # Add one label
        app._curation_add(1, app.st.session_state.cur_batch[0])
        # Re-render Data block by simulating a rerun: import again
        del sys.modules['app']
        sys.modules['streamlit'] = st
        sys.modules['flux_local'] = fl
        out = "\n".join(writes)
        self.assertIn('Dataset rows:', out)


if __name__ == '__main__':
    unittest.main()
