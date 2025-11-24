import os
import sys
import tempfile
import types
import unittest

import numpy as np
from constants import Keys
from tests.helpers.st_streamlit import stub_basic


class TestLastActionAndRowsDisplayOnSave(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("IPO_DATA_ROOT", None)
        for m in ("batch_ui", "streamlit", "flux_local"):
            sys.modules.pop(m, None)

    def test_last_action_and_rows_display_update(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["IPO_DATA_ROOT"] = tmp.name

        st = stub_basic()
        st.session_state.prompt = "last-action"
        st.toast = lambda *a, **k: None  # no-op to keep quiet
        # Minimal state
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4, sigma=1.0, w=np.zeros(4), rng=np.random.default_rng(0)
        )
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl
import batch_ui

        # Call the core save helper directly
        batch_ui._curation_add(+1, np.zeros(4), img=None)
        self.assertIsInstance(st.session_state.get(Keys.LAST_ACTION_TEXT), str)
        self.assertTrue(str(st.session_state.get(Keys.LAST_ACTION_TEXT)).startswith("Saved sample #"))
        self.assertIsInstance(st.session_state.get(Keys.LAST_ACTION_TS), float)
        # Rows display updated to at least 1
        rows_disp = st.session_state.get(Keys.ROWS_DISPLAY)
        self.assertTrue(str(rows_disp).startswith("1"))


if __name__ == "__main__":
    unittest.main()

