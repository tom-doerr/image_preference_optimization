import sys
import types
import unittest


class TestDatasetRowsArtifact(unittest.TestCase):
    def tearDown(self):
        for m in ("app", "persistence", "flux_local", "streamlit"):
            sys.modules.pop(m, None)
    def test_rows_metric_includes_spinner(self):
        from tests.helpers.st_streamlit import stub_with_writes

        st, writes = stub_with_writes()
        # Memory-only rows: seed 7 live labels so sidebar shows 7
        st.session_state.dataset_y = [1] * 7
        try:
from constants import Keys as _K
            st.session_state[_K.DATASET_Y] = st.session_state.dataset_y
        except Exception:
            pass
        sys.modules["streamlit"] = st

        # Stub flux + persistence to avoid heavy imports and control rows
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        p = types.ModuleType("persistence")
        p.state_path_for_prompt = lambda prompt: "latent_state_dummy.npz"
        p.dataset_rows_for_prompt = lambda prompt: 7
        p.get_dataset_for_prompt_or_session = lambda *a, **k: (None, None)
        p.dataset_stats_for_prompt = lambda prompt: {
            "rows": 7,
            "pos": 4,
            "neg": 3,
            "d": 0,
            "recent_labels": [1, -1, 1],
        }
        p.export_state_bytes = lambda *a, **k: b""
        p.read_metadata = lambda path: {
            "app_version": "0",
            "created_at": "",
            "prompt": "p",
        }
        sys.modules["persistence"] = p

        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        out = "\n".join(writes)
        # Sidebar stubs format metrics as "label: value"
        lines = [ln for ln in out.splitlines() if ln.startswith("Dataset rows:")]
        self.assertTrue(lines, "no Dataset rows line found")
        row_line = lines[-1]
        self.assertIn("Dataset rows: 7", row_line)


if __name__ == "__main__":
    unittest.main()
