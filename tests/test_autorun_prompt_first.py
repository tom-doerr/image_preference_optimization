import sys
import types
import unittest

from tests.helpers.st_streamlit import stub_basic


class TestAutorunPromptFirst(unittest.TestCase):
    def test_import_initializes_state_without_prompt_only_image(self):
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_basic(pre_images=False)

        # Minimal flux_local stub; prompt-only helpers are no longer used
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        import app  # noqa: F401
        st = sys.modules['streamlit']
        # App should set up images field but not decode a prompt-only image.
        self.assertTrue('images' in st.session_state)
        self.assertIn(st.session_state.images, (None, (None, None)))
        self.assertTrue('prompt_image' in st.session_state)
        self.assertIsNone(st.session_state.prompt_image)


if __name__ == '__main__':
    unittest.main()

