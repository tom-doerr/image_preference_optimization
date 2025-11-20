import sys
import types
import unittest
import numpy as np


class BatchMultipleSavesTest(unittest.TestCase):
    def tearDown(self):
        for m in (
            "batch_ui",
            "streamlit",
            "latent_logic",
            "persistence",
            "flux_local",
            "value_scorer",
        ):
            sys.modules.pop(m, None)

    def test_multiple_labels_append_rows(self):
        from tests.helpers import st_streamlit

        st, _ = st_streamlit.stub_with_writes()
        st.session_state.prompt = "multi-save"
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4, sigma=1.0, rng=np.random.default_rng(0)
        )
        st.session_state.steps = 3
        st.session_state.guidance_eff = 0.0
        st.session_state.cur_labels = [None, None]
        st.session_state.cur_batch = [np.zeros(4), np.ones(4)]

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        sys.modules["latent_logic"] = ll

        saved = []

        p = types.ModuleType("persistence")
        p.append_dataset_row = lambda prompt, feat, label: len(saved)
        p.save_sample_image = lambda *a, **k: None
        p.get_dataset_for_prompt_or_session = lambda *a, **k: (None, None)
        sys.modules["persistence"] = p

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        sys.modules["flux_local"] = fl

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "ok")
        vs.get_value_scorer = lambda *a, **k: (lambda f: 0.0)
        sys.modules["value_scorer"] = vs

        sys.modules["streamlit"] = st
        import batch_ui

        batch_ui._curation_add(1, st.session_state.cur_batch[0])
        batch_ui._curation_add(-1, st.session_state.cur_batch[1])

        self.assertEqual(len(st.session_state.dataset_y), 2)
        self.assertEqual(st.session_state.dataset_y[0], 1)
        self.assertEqual(st.session_state.dataset_y[1], -1)


if __name__ == "__main__":
    unittest.main()
