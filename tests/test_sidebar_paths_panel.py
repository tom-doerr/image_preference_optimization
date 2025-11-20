import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarPathsPanel(unittest.TestCase):
    def test_paths_panel_shows_state_and_dataset_paths(self):
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

        # Paths panel has been removed to keep the sidebar concise.
        out = "\n".join(writes)
        self.assertNotIn("State path:", out)
        self.assertNotIn("Dataset path:", out)


if __name__ == "__main__":
    unittest.main()
