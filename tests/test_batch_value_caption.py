import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_capture_images


class BatchValueCaptionTest(unittest.TestCase):
    def tearDown(self):
        for m in ("batch_ui", "streamlit", "latent_logic", "persistence", "value_scorer", "flux_local"):
            sys.modules.pop(m, None)

    def test_value_shown_in_image_caption(self):
        st, images = stub_capture_images()
        st.session_state.prompt = "cap-test"
        st.session_state.lstate = types.SimpleNamespace(
            d=4, width=64, height=64, sigma=1.0, rng=np.random.default_rng(0)
        )
        st.session_state.steps = 3
        st.session_state.guidance_eff = 0.0
        st.session_state.cur_labels = [None]
        st.session_state.cur_batch = [np.ones(4)]
        sys.modules["streamlit"] = st

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        ll.z_to_latents = lambda lstate, z: z  # identity for stub
        sys.modules["latent_logic"] = ll

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        sys.modules["flux_local"] = fl

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.5, "ok")
        vs.get_value_scorer = lambda *a, **k: (lambda f: 0.5)
        sys.modules["value_scorer"] = vs

        p = types.ModuleType("persistence")
        p.append_dataset_row = lambda *a, **k: 1
        p.save_sample_image = lambda *a, **k: None
        p.dataset_rows_for_prompt = lambda *a, **k: 0
        sys.modules["persistence"] = p

        import batch_ui

        batch_ui._render_batch_ui()
        # Captured captions should include the value text
        self.assertTrue(any("Value: 0.500" in c for c in images), images)


if __name__ == "__main__":
    unittest.main()
