import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_click_button


class TestGenerateFromPromptButton(unittest.TestCase):
    def test_prompt_only_image_not_used_anymore(self):
        if "app" in sys.modules:
            del sys.modules["app"]
        sys.modules["streamlit"] = stub_click_button("Generate from Prompt")
        # Stub flux_local functions
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image-lat"
        fl.generate_flux_image = lambda *a, **kw: "ok-image-text"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        import app

        # Prompt-only image UI has been removed; prompt_image stays None
        self.assertIsNone(app.st.session_state.prompt_image)


if __name__ == "__main__":
    unittest.main()
