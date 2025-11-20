import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestDatasetViewerPanel(unittest.TestCase):
    def test_dataset_viewer_renders(self):
        st, writes = stub_with_writes()
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **kw: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        out = "\n".join(writes)
        self.assertIn("Viewing dataset:", out)
        self.assertTrue("Rows:" in out or "No datasets found" in out)


if __name__ == "__main__":
    unittest.main()
