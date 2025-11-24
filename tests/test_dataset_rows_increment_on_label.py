import glob
import os
import sys
import tempfile
import types
import unittest

import numpy as np


class DatasetRowsIncrementOnLabelTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("IPO_DATA_ROOT", None)
        for m in (
            "app",
            "app_api",
            "batch_ui",
            "streamlit",
            "latent_logic",
            "flux_local",
            "value_scorer",
            "persistence",
        ):
            sys.modules.pop(m, None)

    def test_click_label_updates_rows_and_disk(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["IPO_DATA_ROOT"] = tmp.name

        from tests.helpers import st_streamlit

        st, writes = st_streamlit.stub_with_writes()
        st.session_state.prompt = "rows-click"
        st.session_state.lstate = types.SimpleNamespace(
            width=64,
            height=64,
            d=4,
            sigma=1.0,
            rng=np.random.default_rng(0),
            w=np.zeros(4),
            mu=np.zeros(4),
            step=0,
        )
        st.session_state.cur_labels = [None]
        st.session_state.cur_batch = [np.ones(4)]
        st.session_state.steps = 1
        st.session_state.guidance_eff = 0.0
        st.session_state.use_fragments = False
        sys.modules["streamlit"] = st

        # Minimal latent/math stubs
        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(getattr(lstate, "d", 0))
        sys.modules["latent_logic"] = ll

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        sys.modules["flux_local"] = fl

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "ok")
        vs.get_value_scorer = lambda *a, **k: (lambda f: 0.0)
        sys.modules["value_scorer"] = vs

        # Minimal latent_opt stub so persistence can import without heavy deps
        lo = types.ModuleType("latent_opt")
        lo.dumps_state = lambda *a, **k: b""
        lo.loads_state = lambda *a, **k: b""
        lo.state_summary = lambda *a, **k: {}
        sys.modules["latent_opt"] = lo

        # Import persistence after IPO_DATA_ROOT is set
        import ipo.core.persistence as persistence  # noqa: F401
import batch_ui
        from ipo.core.persistence import dataset_rows_for_prompt_dim
from constants import Keys

        batch_ui._curation_add(1, np.ones(4))
        batch_ui._curation_add(-1, -np.ones(4))

        rows = dataset_rows_for_prompt_dim(st.session_state.prompt, st.session_state.lstate.d)
        self.assertEqual(rows, 2)
        disp = st.session_state.get(Keys.ROWS_DISPLAY, "")
        self.assertTrue(str(disp).startswith("2"))
        files = glob.glob(os.path.join(tmp.name, "*", "*/sample.npz"))
        self.assertEqual(len(files), 2)


if __name__ == "__main__":
    unittest.main()
