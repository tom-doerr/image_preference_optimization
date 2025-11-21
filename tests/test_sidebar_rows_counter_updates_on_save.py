import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarRowsCounterUpdatesOnSave(unittest.TestCase):
    def test_rows_counter_increments_after_save(self):
        st, writes = stub_with_writes()
        prompt = "rows-counter-save"
        st.text_input = lambda *_, value="": prompt
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **k: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        # First import renders sidebar with zero rows
        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        out0 = "\n".join(writes)
        self.assertIn("Dataset rows:", out0)
        self.assertIn("Dataset rows: 0", out0)

        # Save one labeled row via curation helper
        zs = app.st.session_state.cur_batch
        app._curation_add(1, zs[0])

        # Simulate rerun
        del sys.modules["app"]
        sys.modules["streamlit"] = st
        sys.modules["flux_local"] = fl
        import app as app2  # noqa: F401

        out1 = "\n".join(writes)
        self.assertIn("Dataset rows: 1", out1)
        self.assertTrue("Rows (disk)" in out1)


if __name__ == "__main__":
    unittest.main()

