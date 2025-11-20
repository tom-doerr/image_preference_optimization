import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestCurationMode(unittest.TestCase):
    def test_accept_reject_updates_dataset(self):
        st = stub_basic(pre_images=False)

        # Enable curation mode and use small batch
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if "Batch curation mode" in label else value

            @staticmethod
            def slider(label, *a, **k):
                return 3 if "Batch size" in label else (k.get("value") or 1)

        st.sidebar = SB()
        sys.modules["streamlit"] = st

        # Stub flux_local to avoid heavy work
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **kw: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        import app

        # Simulate accepting first, rejecting second
        z_list = app.st.session_state.cur_batch
        self.assertEqual(len(z_list), 3)
        app._curation_add(1, z_list[0])
        app._curation_add(-1, z_list[1])
        X = app.st.session_state.dataset_X
        y = app.st.session_state.dataset_y
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(int(y[0] + y[1]), 0)  # one +1 and one -1


if __name__ == "__main__":
    unittest.main()
