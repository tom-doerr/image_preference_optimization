import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_with_writes


class SavedPathToastTest(unittest.TestCase):
    def tearDown(self):
        for m in ("batch_ui", "streamlit", "latent_logic", "persistence"):
            sys.modules.pop(m, None)

    def test_saved_path_appended_to_toast(self):
        st, writes = stub_with_writes()
        st.toast = lambda msg: st.sidebar.write(str(msg))
        sys.modules["streamlit"] = st

        # Stub prompt path
        p = types.ModuleType("persistence")
        p.append_dataset_row = lambda *a, **k: 42
        p.save_sample_image = lambda *a, **k: None
        p.dataset_rows_for_prompt = lambda prompt: 0
        p.data_root_for_prompt = lambda prompt: "data/abc123"
        sys.modules["persistence"] = p

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(4)
        sys.modules["latent_logic"] = ll

        st.session_state.prompt = "p"
        st.session_state.lstate = types.SimpleNamespace(d=4, width=64, height=64, sigma=1.0, rng=np.random.default_rng(0))
        st.session_state.dataset_y = np.zeros((0,))
        st.session_state.dataset_X = np.zeros((0, 4))

        import batch_ui

        batch_ui._curation_add(+1, np.zeros(4))
        out = "\n".join(writes)
        self.assertIn("Saved sample #42 → data/abc123/000042", out)


if __name__ == "__main__":
    unittest.main()
