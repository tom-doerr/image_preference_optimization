import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestStepScoresTruncation(unittest.TestCase):
    def test_prints_first_8_and_4_tiles(self):
        st, writes = stub_with_writes()
        st.session_state.prompt = "steps-truncate"
        st.session_state.iter_steps = 10
        st.session_state.iter_eta = 0.0

        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, *a, **k):
                if "Generation mode" in label:
                    return "Batch curation"
                if "Value model" in label:
                    return "Ridge"
                if label == "Model":
                    return "stabilityai/sd-turbo"
                return k.get("value") or (a[0][0] if a and a[0] else None)

        st.sidebar = SB()
        # Keep capture hooks
        st.sidebar.write = lambda *a, **k: writes.append(str(a[0]) if a else "")
        st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.set_model = lambda *a, **k: None
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        import app as appmod

        # Force non-zero weights so scores are not all zeros
        d = appmod.st.session_state.lstate.d
        appmod.st.session_state.lstate.w = __import__("numpy").ones(d)

        # Re-import to trigger sidebar render again with weights set
        writes.clear()
        del sys.modules["app"]
        sys.modules["streamlit"] = st
        sys.modules["flux_local"] = fl
        import app  # noqa: F401

        out = "\n".join(writes)
        # 1) The compact line should include exactly 8 comma-separated values
        line = next((ln for ln in writes if ln.startswith("Step scores:")), "")
        self.assertTrue(line.startswith("Step scores:"))
        num_vals = len(line.split(":", 1)[1].split(","))
        self.assertEqual(num_vals, 8)
        # 2) Only tiles for first 4 steps should be present
        self.assertIn("Step 1:", out)
        self.assertIn("Step 2:", out)
        self.assertIn("Step 3:", out)
        self.assertIn("Step 4:", out)
        self.assertNotIn("Step 5:", out)


if __name__ == "__main__":
    unittest.main()
