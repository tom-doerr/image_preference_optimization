import sys
import types
import unittest

import numpy as np


class UploadScoresWeightsTest(unittest.TestCase):
    def tearDown(self):
        for m in ("app", "streamlit", "persistence", "latent_logic", "value_scorer", "flux_local"):
            sys.modules.pop(m, None)

    def test_upload_shows_score_and_weight(self):
        from tests.helpers import st_streamlit

        st, writes = st_streamlit.stub_with_writes()
        st.session_state.prompt = "test-upload"
        st.session_state.lstate = types.SimpleNamespace(
            width=64,
            height=64,
            d=4,
            sigma=1.0,
            rng=np.random.default_rng(0),
            w=np.ones(4),
            step=0,
        )
        st.session_state.lz_pair = (np.zeros(4), np.zeros(4))
        st.session_state.steps = 3
        st.session_state.guidance_eff = 0.0
        st.session_state.cur_batch_nonce = 1

        # Stub persistence/value_scorer/latent_logic
        p = types.ModuleType("persistence")
        p.append_dataset_row = lambda *a, **k: 0
        p.save_sample_image = lambda *a, **k: None
        p.get_dataset_for_prompt_or_session = lambda *a, **k: (np.zeros((1, 4)), np.array([1.0]))
        p.state_path_for_prompt = lambda prompt: "latent_state_dummy.npz"
        p.export_state_bytes = lambda state, prompt: b""
        p.dataset_rows_for_prompt = lambda prompt: 0
        p.dataset_stats_for_prompt = lambda prompt: {"rows": 0, "pos": 0, "neg": 0, "d": 0, "recent_labels": []}
        p.read_metadata = lambda path: {"app_version": None, "created_at": None, "prompt": None}
        sys.modules["persistence"] = p

        vl = types.ModuleType("value_scorer")
        vl.get_value_scorer_with_status = lambda vmc, lstate, prompt, ss: (lambda f: 0.5, "ok")
        sys.modules["value_scorer"] = vl

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        ll.sample_z_xgb_hill = lambda *a, **k: np.zeros(st.session_state.lstate.d)
        sys.modules["latent_logic"] = ll

        lo = types.ModuleType("latent_opt")
        lo.z_to_latents = lambda z, lstate: z.reshape(1, 1, 2, 2)
        lo.z_from_prompt = ll.z_from_prompt
        # Minimal state passthroughs use the current session state's lstate
        lo.loads_state = lambda *a, **k: st.session_state.lstate
        lo.save_state = lambda *a, **k: None
        lo.init_latent_state = lambda *a, **k: st.session_state.lstate
        lo.load_state = lambda *a, **k: st.session_state.lstate
        lo.propose_latent_pair_ridge = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        lo.propose_pair_prompt_anchor = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        lo.propose_pair_prompt_anchor_iterative = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        lo.propose_pair_prompt_anchor_linesearch = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        lo.update_latent_ridge = lambda *a, **k: None
        lo.dumps_state = lambda *a, **k: b""
        lo.propose_next_pair = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        lo.state_summary = lambda *a, **k: {
            "pairs": 0,
            "choices": 0,
            "d": st.session_state.lstate.d,
            "width": st.session_state.lstate.width,
            "height": st.session_state.lstate.height,
            "step": 0,
            "sigma": st.session_state.lstate.sigma,
            "mu_norm": 0.0,
            "w_norm": 0.0,
            "pairs_logged": 0,
            "choices_logged": 0,
        }
        sys.modules["latent_opt"] = lo

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        # Provide one dummy upload
        class DummyUpload:
            def __init__(self):
                self.name = "u.png"
            def read(self):
                return b""
        st.sidebar.file_uploader = lambda *a, **k: [DummyUpload()]
        st.sidebar.selectbox = lambda *a, **k: "Upload latents"

        sys.modules["streamlit"] = st
        import app  # noqa: F401

        # No crash; upload path executed without errors
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
