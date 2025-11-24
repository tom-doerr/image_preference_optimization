import sys
import types
import unittest

import numpy as np

from tests.helpers.st_streamlit import stub_capture_images


class TestXgbHillStepsParam(unittest.TestCase):
    def tearDown(self):
        for m in ("batch_ui", "streamlit", "flux_local", "latent_logic", "value_scorer"):
            sys.modules.pop(m, None)

    def test_batch_uses_iter_steps_for_xgb_hill(self):
        # Arrange minimal Streamlit + session state
        st, _images = stub_capture_images()
        st.session_state.vm_choice = "XGBoost"
        st.session_state.steps = 1
        st.session_state.guidance_eff = 0.0
        st.session_state.iter_steps = 7  # <- what the batch should pass through
        st.session_state.lr_mu_ui = 0.3
        st.session_state.trust_r = None
        # Minimal latent state and batch
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4, sigma=1.0, w=np.ones(4), rng=np.random.default_rng(0)
        )
        st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
        st.session_state.cur_labels = [None, None]

        # Stub modules used by batch_ui
        sys.modules["streamlit"] = st

        # Capture the steps argument received by sample_z_xgb_hill
        seen = {}

        def _sample_z_xgb_hill(_state, _prompt, _scorer, *, steps=3, step_scale=0.2, trust_r=None):
            seen["steps"] = steps
            # return a deterministic latent
            return np.zeros(_state.d)

        ll = types.ModuleType("latent_logic")
        ll.sample_z_xgb_hill = _sample_z_xgb_hill
        ll.z_from_prompt = lambda s, p: np.zeros(s.d)
        ll.z_to_latents = lambda *_a, **_k: np.zeros((1, 1, 1, 1))
        sys.modules["latent_logic"] = ll

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        sys.modules["flux_local"] = fl

        vs = types.ModuleType("value_scorer")
        # Provide an OK scorer so the XGB path activates
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "ok")
        sys.modules["value_scorer"] = vs

        # Act
import batch_ui

        batch_ui._render_batch_ui()

        # Assert
        self.assertEqual(seen.get("steps"), 7, "iter_steps should pass to sample_z_xgb_hill")


if __name__ == "__main__":
    unittest.main()

