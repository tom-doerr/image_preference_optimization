import io
import sys
import types
import unittest
import numpy as np
from contextlib import redirect_stdout
from tests.helpers.st_streamlit import stub_basic


class RowsCliPrintTest(unittest.TestCase):
    def tearDown(self):
        for m in ("batch_ui", "streamlit", "latent_logic", "persistence"):
            sys.modules.pop(m, None)

    def test_rows_prints_live_and_disp(self):
        st = stub_basic()
        st.session_state.prompt = "rows-cli"
        st.session_state.lstate = types.SimpleNamespace(
            d=4, width=64, height=64, sigma=1.0, rng=np.random.default_rng(0)
        )
from constants import Keys

        st.session_state.dataset_y = [0.0]
        st.session_state.dataset_X = [np.zeros(4)]
        st.session_state[Keys.DATASET_Y] = st.session_state.dataset_y
        st.session_state[Keys.DATASET_X] = st.session_state.dataset_X

        # Stub latent_logic and persistence
        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        sys.modules["latent_logic"] = ll

        p = types.ModuleType("persistence")
        # New code path uses append_sample wrapper
        p.append_dataset_row = lambda *a, **k: 1
        p.save_sample_image = lambda *a, **k: None
        p.append_sample = lambda *a, **k: 2
        p.dataset_rows_for_prompt = lambda prompt: 2
        sys.modules["persistence"] = p

        sys.modules["streamlit"] = st
import batch_ui

        buf = io.StringIO()
        with redirect_stdout(buf):
            batch_ui._curation_add(+1, np.zeros(4))
        out = buf.getvalue() + "\n".join(st._write_captured)
        # Memory-only rows: display mirrors live count; expect disp not disk
        self.assertIn("[rows] live=2 disp=2", out)


if __name__ == "__main__":
    unittest.main()
