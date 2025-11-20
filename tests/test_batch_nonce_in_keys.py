import sys
import types
import unittest
import numpy as np


class BatchNonceKeysTest(unittest.TestCase):
    def tearDown(self):
        for m in (
            "batch_ui",
            "streamlit",
            "latent_logic",
            "persistence",
            "value_scorer",
            "flux_local",
        ):
            sys.modules.pop(m, None)

    def test_keys_include_nonce(self):
        from tests.helpers import st_streamlit

        pressed = []

        def button(label, *a, **k):
            if "key" in k:
                pressed.append(k["key"])
            return False

        st, _ = st_streamlit.stub_with_writes()
        st.button = button
        st.sidebar.button = button
        st.session_state.prompt = "nonce-keys"
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4, sigma=1.0, rng=np.random.default_rng(0)
        )
        st.session_state.steps = 3
        st.session_state.guidance_eff = 0.0
        st.session_state.cur_labels = [None, None]
        st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
        st.session_state.cur_batch_nonce = 7  # simulate existing batch nonce

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        ll.z_to_latents = lambda lstate, z: np.zeros((1, 1, 2, 2))
        ll.sample_z_xgb_hill = lambda *a, **k: np.zeros(st.session_state.lstate.d)
        sys.modules["latent_logic"] = ll

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "ok")
        vs.get_value_scorer = lambda *a, **k: (lambda f: 0.0)
        sys.modules["value_scorer"] = vs

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        sys.modules["flux_local"] = fl

        p = types.ModuleType("persistence")
        p.append_dataset_row = lambda *a, **k: 0
        p.save_sample_image = lambda *a, **k: None
        p.get_dataset_for_prompt_or_session = lambda *a, **k: (None, None)
        sys.modules["persistence"] = p

        sys.modules["streamlit"] = st
        import batch_ui

        batch_ui._render_batch_ui()

        # Keys should contain the batch nonce so Good/Bad buttons are unique across batches.
        self.assertTrue(
            any(f"_{st.session_state.cur_batch_nonce}_" in k for k in pressed)
        )


if __name__ == "__main__":
    unittest.main()
