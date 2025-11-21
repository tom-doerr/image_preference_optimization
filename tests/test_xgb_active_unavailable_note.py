import sys
import types
import unittest

from tests.helpers import st_streamlit


class XGBActiveUnavailableTest(unittest.TestCase):
    def tearDown(self):
        for name in (
            "streamlit",
            "app",
            "persistence",
            "value_scorer",
            "latent_logic",
            "batch_ui",
            "flux_local",
            "modes",
        ):
            sys.modules.pop(name, None)

    def test_sidebar_shows_xgb_active_no_when_unavailable(self):
        st, writes = st_streamlit.stub_with_writes()
        st.session_state.clear()
        st.session_state.prompt = "xgb-active-unavailable"
        st.session_state.vm_choice = "XGBoost"
        st.session_state.reg_lambda = 1e-3
        from latent_opt import init_latent_state

        st.session_state.lstate = init_latent_state(width=16, height=16, seed=0)
        st.session_state.lz_pair = (
            st.session_state.lstate.mu,
            st.session_state.lstate.mu,
        )
        sys.modules["streamlit"] = st

        # Minimal stubs
        p = types.ModuleType("persistence")
        p.state_path_for_prompt = lambda prompt: "latent_state_dummy.npz"
        p.dataset_rows_for_prompt = lambda prompt: 0
        p.get_dataset_for_prompt_or_session = lambda prompt, ss: (None, None)
        p.read_metadata = lambda path: {
            "app_version": None,
            "created_at": None,
            "prompt": None,
        }
        p.dataset_stats_for_prompt = lambda prompt: {
            "rows": 0,
            "pos": 0,
            "neg": 0,
            "d": 0,
            "recent_labels": [],
        }
        p.export_state_bytes = lambda state, prompt: b""
        sys.modules["persistence"] = p

        vs = types.ModuleType("value_scorer")
        # Unified API: return (None, status) to indicate unavailable
        vs.get_value_scorer = lambda *a, **k: (None, "xgb_unavailable")
        # Keep shim symbol for tests that import it, but it won't be used here
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "xgb_unavailable")
        sys.modules["value_scorer"] = vs

        m = types.ModuleType("modes")
        m.run_mode = lambda *a, **k: None
        sys.modules["modes"] = m

        f = types.ModuleType("flux_local")
        f.generate_flux_image_latents = lambda *a, **k: "img"
        f.set_model = lambda *a, **k: None
        f.get_last_call = lambda: {}
        f.generate_flux_image = lambda *a, **k: "img"
        sys.modules["flux_local"] = f

        import app  # noqa: F401

        # Assert the active line reflects readiness (no when unavailable)
        text = "\n".join(str(w) for w in writes)
        self.assertIn("Value scorer status: xgb_unavailable", text)
        self.assertIn("XGBoost active: no", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

