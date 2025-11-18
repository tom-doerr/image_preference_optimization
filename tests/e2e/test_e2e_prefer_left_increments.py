import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_main_writes


class TestE2EPreferLeftIncrements(unittest.TestCase):
    def test_click_prefer_left_increments_iterations(self):
        st, writes = stub_with_main_writes(pre_images=False)
        # Use a unique prompt so no on-disk state collides with different dims
        st.text_input = lambda *_, value="": 'e2e prefer left'
        # Ensure sliders return positional default when used positionally
        st.slider = lambda *args, **kwargs: kwargs.get('value', args[2] if len(args) >= 3 else 1.0)

        # Click handler that returns True only for 'Prefer Left' once
        clicked = {'left': False}
        def button(label, *a, **k):
            if label == 'Prefer Left' and not clicked['left']:
                clicked['left'] = True
                return True
            return False
        st.button = button

        sys.modules['streamlit'] = st
        # Ensure a fresh in-memory state (avoid loading any on-disk NPZ)
        import os as _os
        _os.path.exists = lambda *_a, **_k: False
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app
        # Minimal e2e: emulate one preference via the app's public function
        z_a, z_b = app.st.session_state.lz_pair
        # Ensure a clean history to avoid shape mismatch with legacy files
        app.st.session_state.lstate.mu_hist = None
        app.update_latent_ridge(app.st.session_state.lstate, z_a, z_b, 'a', lr_mu=0.3,
                                 feats_a=(z_a - app.z_from_prompt(app.st.session_state.lstate, app.st.session_state.prompt)),
                                 feats_b=(z_b - app.z_from_prompt(app.st.session_state.lstate, app.st.session_state.prompt)))
        self.assertEqual(app.st.session_state.lstate.step, 1)


if __name__ == '__main__':
    unittest.main()
