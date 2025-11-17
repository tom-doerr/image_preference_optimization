import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarRidgeOnly(unittest.TestCase):
    def test_ridge_only_note_present(self):
        st, writes = stub_with_writes()
        sys.modules['streamlit'] = st
        # Stub flux_local so import works
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.generate_flux_image = lambda *a, **kw: 'ok-image-text'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app  # noqa: F401
        if 'lstate' not in st.session_state:
            st.session_state.lstate = app.init_latent_state()
            st.session_state.lz_pair = app.propose_latent_pair_ridge(st.session_state.lstate)
        self.assertIn('lstate', st.session_state)
        self.assertIn('lz_pair', st.session_state)


if __name__ == '__main__':
    unittest.main()
