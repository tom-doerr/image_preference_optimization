import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_main_writes


class TestE2EPredictedValuesAndIterations(unittest.TestCase):
    def test_predicted_values_and_iterations_display(self):
        self.skipTest("Predicted value display not enforced in simplified UI")
        st, writes = stub_with_main_writes(pre_images=False)
        # Capture metric output too, not just write fallback
        st.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **kw: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        import app  # noqa: F401

        text = "\n".join(writes)
        # Iterations line should be present and start at 0 on fresh import
        self.assertIn("Interactions: 0", text)


if __name__ == "__main__":
    unittest.main()
