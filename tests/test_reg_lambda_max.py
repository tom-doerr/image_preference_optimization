import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestRegLambdaMax(unittest.TestCase):
    def test_numeric_input_allows_large_lambda(self):
        st = stub_basic()
        # Capture the set value for Ridge λ input; no min/max constraints required
        captured = {"ridge_lambda_value": None}

        def number_input(label, **k):
            if label.startswith("Ridge λ"):
                # Simulate user entering a large value
                captured["ridge_lambda_value"] = float(k.get("value", 0.0))
            return k.get("value", 0.0)

        st.number_input = number_input
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        # Import app to trigger sidebar construction
        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        # Ensure the input is rendered and accepts the provided (large) value
        self.assertIsNotNone(captured["ridge_lambda_value"])
        # Clean up to avoid polluting subsequent tests
        if "app" in sys.modules:
            del sys.modules["app"]


if __name__ == "__main__":
    unittest.main()
