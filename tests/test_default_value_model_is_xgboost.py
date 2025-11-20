import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestDefaultValueModelIsXGBoost(unittest.TestCase):
    def test_default_vm_choice_xgboost(self):
        st, writes = stub_with_writes()
        # Use the default selectbox from the stub, which returns an unrelated string.
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl
        if "app" in sys.modules:
            del sys.modules["app"]
        import app

        self.assertEqual(app.st.session_state.get("vm_choice"), "XGBoost")
        out = "\n".join(writes)
        self.assertIn("Value model: XGBoost", out)


if __name__ == "__main__":
    unittest.main()
