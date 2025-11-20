import sys
import types
import unittest
import numpy as np

from tests.helpers.st_streamlit import stub_basic


class TestBatchBestOfMode(unittest.TestCase):
    def test_choose_marks_one_good_rest_bad(self):
        # Stub Streamlit and enable Batch mode
        st = stub_basic()

        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                # Enable Batch curation; best-of toggle set via session_state below
                return True if "Batch curation mode" in label else value

            @staticmethod
            def slider(label, *a, **k):
                if "Batch size" in label:
                    return 3
                return k.get("value", 1)

        st.sidebar = SB()
        sys.modules["streamlit"] = st

        # Fast flux_local stub
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        # Import app to build initial batch
        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        # Enable best-of mode and capture current batch
        st.session_state.batch_best_of = True
        before = [np.copy(z) for z in st.session_state.cur_batch]

        # Click "Choose 0" on rerun
        def _btn(label, *a, **k):
            return label == "Choose 0"

        st.button = _btn
        if "app" in sys.modules:
            del sys.modules["app"]
        import app as app2  # noqa: F401

        # After choice, dataset_y should contain one +1 and the rest -1
        y = st.session_state.get("dataset_y")
        self.assertIsNotNone(y)
        self.assertEqual(len(y), len(before))
        self.assertEqual(float(y[0]), 1.0)
        self.assertTrue(all(float(y[j]) == -1.0 for j in range(1, len(before))))


if __name__ == "__main__":
    unittest.main()
