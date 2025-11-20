import sys
import types
import unittest


class DatasetRowsLiveTest(unittest.TestCase):
    def tearDown(self):
        for m in (
            "app",
            "streamlit",
            "persistence",
            "latent_logic",
            "flux_local",
            "value_scorer",
        ):
            sys.modules.pop(m, None)

    def test_dataset_rows_prefers_live_session(self):
        from tests.helpers import st_streamlit

        st, writes = st_streamlit.stub_with_writes()
        st.session_state.prompt = "rows-live"
        st.session_state.dataset_y = [1, 1, -1]
        st.session_state.dataset_X = []
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4, rng=None, sigma=1.0, w=0, step=0, mu=0
        )
        st.session_state.lz_pair = (0, 0)

        p = types.ModuleType("persistence")
        p.state_path_for_prompt = lambda prompt: "latent_state_dummy.npz"
        p.dataset_rows_for_prompt = lambda prompt: 1
        p.dataset_stats_for_prompt = lambda prompt: {
            "rows": 1,
            "pos": 1,
            "neg": 0,
            "d": 4,
            "recent_labels": [],
        }
        p.export_state_bytes = lambda state, prompt: b""
        p.read_metadata = lambda path: {
            "app_version": None,
            "created_at": None,
            "prompt": None,
        }
        p.get_dataset_for_prompt_or_session = lambda prompt, ss: (None, None)
        sys.modules["persistence"] = p

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: 0
        ll.z_to_latents = lambda *a, **k: 0
        ll.sample_z_xgb_hill = lambda *a, **k: 0
        ll.propose_latent_pair_ridge = lambda *a, **k: (0, 0)
        ll.propose_pair_prompt_anchor = lambda *a, **k: (0, 0)
        ll.propose_pair_prompt_anchor_iterative = lambda *a, **k: (0, 0)
        ll.propose_pair_prompt_anchor_linesearch = lambda *a, **k: (0, 0)
        ll.update_latent_ridge = lambda *a, **k: None
        sys.modules["latent_logic"] = ll

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "ok")
        vs.get_value_scorer = lambda *a, **k: (lambda f: 0.0)
        sys.modules["value_scorer"] = vs

        sys.modules["streamlit"] = st
        import app  # noqa: F401

        # Dataset rows should reflect live length (3 > disk 1)
        self.assertTrue(any("Dataset rows: 3" in w for w in writes))


if __name__ == "__main__":
    unittest.main()
