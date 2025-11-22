import sys
import types
import unittest

from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarDatasetHints(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar", "streamlit", "persistence", "flux_local"):
            sys.modules.pop(m, None)

    def test_dataset_path_and_disk_rows_and_dim_hint(self):
        st, writes = stub_with_writes()
        st.session_state.prompt = "dim-hint"
        # current latent dim (e.g., 25600)
        st.session_state.lstate = types.SimpleNamespace(d=25600)
        sys.modules["streamlit"] = st

        # Stub persistence: say there are 5 rows on disk with a smaller feature dim
        p = types.ModuleType("persistence")
        p.dataset_rows_for_prompt = lambda prompt: 5
        # Return an array-like with shape (5, 9216)
        p.get_dataset_for_prompt_or_session = lambda prompt, ss: (types.SimpleNamespace(shape=(5, 9216)), None)
        p.read_metadata = lambda path: {}
        p.dataset_stats_for_prompt = lambda prompt: {"pos": 0, "neg": 0, "d": 0}
        sys.modules["persistence"] = p

        sys.modules["flux_local"] = types.SimpleNamespace(set_model=lambda *a, **k: None)

        import ipo.ui.ui_sidebar as u
        u.render_sidebar_tail(
            st,
            st.session_state.lstate,
            st.session_state.prompt,
            "latent_state_test.npz",
            "XGBoost",
            0,
            0.0,
            "stabilityai/sd-turbo",
            lambda *a, **k: None,
            lambda *a, **k: None,
        )

        text = "\n".join(str(w) for w in writes)
        self.assertIn("Dataset path:", text)
        self.assertIn("Rows (disk): 5", text)
        self.assertIn("Dataset recorded at d=9216; current d=25600", text)


if __name__ == "__main__":
    unittest.main()
