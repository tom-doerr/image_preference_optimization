import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestRegLambdaNumberInput(unittest.TestCase):
    def test_number_input_overrides_slider(self):
        st = stub_basic(pre_images=False)

        # Force batch mode on so training UI wires up
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if "Batch curation mode" in label else value

            @staticmethod
            def slider(label, *a, **k):
                # Return default for everything (including Ridge slider)
                return k.get("value", 1)

        st.sidebar = SB()

        # Simulate user typing a precise lambda value in number_input
        st.number_input = lambda *a, **k: 0.037

        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **kw: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        if "app" in sys.modules:
            del sys.modules["app"]
        import app

        # Prepare two labels to trigger a train step
        zs = app.st.session_state.cur_batch
        app._curation_add(1, zs[0])
        app._curation_add(-1, zs[1])

        # Capture lambda passed to ridge_fit
        import latent_logic as ll

        captured = {}

        def _rf(X, y, lam):
            captured["lam"] = lam
            import numpy as np

            return np.zeros(X.shape[1], dtype=float)

        ll.ridge_fit = _rf

        app._curation_train_and_next()
        self.assertAlmostEqual(captured.get("lam", 0.0), 0.037, places=9)


if __name__ == "__main__":
    unittest.main()
