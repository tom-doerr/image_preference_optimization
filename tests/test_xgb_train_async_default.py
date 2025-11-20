import sys
import types
import unittest

from tests.helpers import st_streamlit


class XGBAsyncDefaultTest(unittest.TestCase):
    def tearDown(self):
        for name in (
            "streamlit",
            "app",
            "persistence",
            "value_scorer",
            "latent_logic",
            "batch_ui",
            "flux_local",
        ):
            sys.modules.pop(name, None)

    def test_async_training_default_true(self):
        for name in ("persistence", "value_scorer", "latent_logic", "batch_ui"):
            sys.modules.pop(name, None)
        st = st_streamlit.stub_basic()
        st.session_state.clear()
        st.session_state.prompt = "async-default-test"
        sys.modules["streamlit"] = st
        # Minimal persistence stub to avoid loading real datasets during import
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
        # Minimal flux_local stub to avoid loading torch/diffusers during import
        f = types.ModuleType("flux_local")
        f.generate_flux_image_latents = lambda *a, **k: "img"
        f.set_model = lambda *a, **k: None
        f.get_last_call = lambda: {}
        f.generate_flux_image = lambda *a, **k: "img"
        sys.modules["flux_local"] = f

        import app  # noqa: F401

        self.assertTrue(st.session_state.get("xgb_train_async"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
