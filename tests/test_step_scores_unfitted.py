import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestStepScoresUnfitted(unittest.TestCase):
    def test_shows_na_when_unfitted(self):
        st, writes = stub_with_writes()
        st.session_state.prompt = "step-scores-unfitted"
        # No dataset, default w≈0; choose XGBoost so scorer is unavailable
        st.sidebar.selectbox = staticmethod(lambda *a, **k: "XGBoost")
        sys.modules["streamlit"] = st
        # Minimal flux stub
        fl = types.ModuleType("flux_local")
        fl.set_model = lambda *a, **k: None
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        sys.modules["flux_local"] = fl

        out = "\n".join(writes)
        self.assertIn("Step scores: n/a", out)


if __name__ == "__main__":
    unittest.main()
