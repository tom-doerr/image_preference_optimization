import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_capture_images


class TestImageCaptionsPromptDistance(unittest.TestCase):
    def test_left_right_captions_include_prompt_distance(self):
        st, images = stub_capture_images()
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        import app  # noqa: F401

        joined = "\n".join(images)
        self.assertIn("Left (d_prompt=", joined)
        self.assertIn("Right (d_prompt=", joined)


if __name__ == "__main__":
    unittest.main()
