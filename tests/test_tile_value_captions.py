import sys
import types
import unittest

import numpy as np

from tests.helpers.st_streamlit import stub_capture_images


class TestTileValueCaptions(unittest.TestCase):
    def tearDown(self):
        for m in ("batch_ui", "streamlit", "flux_local", "value_scorer"):
            sys.modules.pop(m, None)

    def _setup_minimal(self):
        st, images = stub_capture_images()
        st.session_state.vm_choice = "XGBoost"
        st.session_state.steps = 1
        st.session_state.guidance_eff = 0.0
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4, sigma=1.0, w=np.zeros(4), rng=np.random.default_rng(0)
        )
        # Minimal batch (2 tiles)
        st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
        st.session_state.cur_labels = [None, None]

        sys.modules["streamlit"] = st

        # Stub flux_local decoder
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl
        return st, images

    def test_captions_show_na_when_scorer_unavailable(self):
        st, images = self._setup_minimal()
        # Scorer returns not ready
        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "xgb_unavailable")
        sys.modules["value_scorer"] = vs

        import batch_ui

        batch_ui._render_batch_ui()
        # Expect captions contain "Value: n/a"
        self.assertTrue(any("Value: n/a" in cap for cap in images))

    def test_captions_show_number_when_scorer_ok(self):
        st, images = self._setup_minimal()
        # Single-scorer rule: provide XGB cache and scorer
        xv = types.ModuleType("xgb_value")
        xv.score_xgb_proba = lambda mdl, f: 0.1234
        sys.modules["xgb_value"] = xv
        st.session_state.xgb_cache = {"model": object(), "n": 1}
        vs = types.ModuleType("value_scorer")
        def _xgb_scorer_with_status(*a, **k):
            # Defer to xgb_value so batch_ui picks it up
            return (lambda f: xv.score_xgb_proba("stub", f), "ok")
        vs.get_value_scorer_with_status = _xgb_scorer_with_status
        sys.modules["value_scorer"] = vs

        import batch_ui

        batch_ui._render_batch_ui()
        # Expect captions contain Value: 0.123
        self.assertTrue(any("Value: 0.123" in cap for cap in images))


if __name__ == "__main__":
    unittest.main()
