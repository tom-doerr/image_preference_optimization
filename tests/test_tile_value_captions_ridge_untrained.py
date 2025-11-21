import sys
import types
import unittest

import numpy as np

from tests.helpers.st_streamlit import stub_capture_images


class TestTileValueCaptionsRidgeUntrained(unittest.TestCase):
    def tearDown(self):
        for m in ("batch_ui", "streamlit", "flux_local", "value_scorer"):
            sys.modules.pop(m, None)

    def test_ridge_shows_zero_value_when_untrained(self):
        st, images = stub_capture_images()
        st.session_state.vm_choice = "Ridge"
        st.session_state.steps = 1
        st.session_state.guidance_eff = 0.0
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4, sigma=1.0, w=np.zeros(4), rng=np.random.default_rng(0)
        )
        st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
        st.session_state.cur_labels = [None, None]

        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        # Use real value_scorer, which will report ridge_untrained, but scorer is usable
        import value_scorer as vs
        sys.modules["value_scorer"] = vs

        import batch_ui

        batch_ui._render_batch_ui()
        self.assertTrue(any("Value: 0.000" in cap for cap in images))


if __name__ == "__main__":
    unittest.main()

