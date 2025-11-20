import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_capture_images


class AsyncQueueValueCaptionTest(unittest.TestCase):
    def tearDown(self):
        for m in ("queue_ui", "streamlit", "latent_state", "batch_ui", "flux_local", "value_scorer", "latent_logic"):
            sys.modules.pop(m, None)

    def test_queue_item_shows_value_in_caption(self):
        st, images = stub_capture_images()
        st.session_state.prompt = "q-cap"
        import latent_state

        st.session_state.lstate = latent_state.init_latent_state()
        st.session_state.steps = 3
        st.session_state.guidance_eff = 0.0
        st.session_state.queue = [
            {
                "z": np.ones(st.session_state.lstate.d),
                "future": type("Fut", (), {"result": lambda self=None: "img"})(),
                "label": None,
            }
        ]
        st.session_state.use_fragments = False
        st.session_state.cur_labels = [None]

        # Minimal helpers
        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(st.session_state.lstate.d)
        ll.z_to_latents = lambda lstate, z, noise_gamma=0.35: z
        ll.propose_latent_pair_ridge = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        ll.propose_pair_prompt_anchor = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        ll.propose_pair_prompt_anchor_iterative = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        ll.propose_pair_prompt_anchor_linesearch = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        ll.update_latent_ridge = lambda *a, **k: None
        sys.modules["latent_logic"] = ll

        bu = types.ModuleType("batch_ui")
        bu._curation_add = lambda lbl, z, img=None: None
        bu._curation_train_and_next = lambda: None
        sys.modules["batch_ui"] = bu

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        sys.modules["flux_local"] = fl

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.25, "ok")
        vs.get_value_scorer = lambda *a, **k: (lambda f: 0.25)
        sys.modules["value_scorer"] = vs

        sys.modules["streamlit"] = st
        import queue_ui

        queue_ui._render_queue_ui()
        self.assertTrue(any("Value: 0.250" in c for c in images), images)


if __name__ == "__main__":
    unittest.main()
