import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestPromptDistanceSidebar(unittest.TestCase):
    def test_sidebar_shows_prompt_distances(self):
        st, writes = stub_with_writes()
        sys.modules['streamlit'] = st
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app  # noqa: F401
        # Render the vector sidebar explicitly to capture writes in this stubbed env
        from ui import render_pair_sidebar
        za, zb = app.st.session_state.lz_pair
        render_pair_sidebar(app.st.session_state.lstate, app.st.session_state.prompt, za, zb, lr_mu_val=0.3)
        text = "\n".join(writes)
        self.assertIn("‖μ−z_prompt‖:", text)
        self.assertIn("‖z_a−z_prompt‖:", text)
        self.assertIn("‖z_b−z_prompt‖:", text)


if __name__ == '__main__':
    unittest.main()
