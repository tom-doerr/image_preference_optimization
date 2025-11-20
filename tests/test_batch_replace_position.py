import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestBatchReplacePosition(unittest.TestCase):
    def test_replace_only_labeled_position(self):
        st = stub_basic(pre_images=False)
        prompt = "batch replace position test"
        st.text_input = lambda *_, value="": prompt

        # Enable batch mode with batch size 2
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if "Batch curation mode" in label else value

            @staticmethod
            def slider(label, *a, **k):
                if "Batch size" in label:
                    return 2
                return k.get("value", 1)

        st.sidebar = SB()
        sys.modules["streamlit"] = st

        # Fast stubs
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **k: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        import app

        zs = list(app.st.session_state.cur_batch)
        # Replace position 0
        app._curation_replace_at(0)
        z2 = list(app.st.session_state.cur_batch)
        # Position 0 changes, position 1 remains
        assert (zs[0] != z2[0]).any()
        assert (zs[1] == z2[1]).all()


if __name__ == "__main__":
    unittest.main()
