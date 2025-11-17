import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_click_button


class TestGenerateFromPromptButton(unittest.TestCase):
    def test_button_generates_prompt_image(self):
        sys.modules['streamlit'] = stub_click_button('Generate from Prompt')
        # Stub flux_local functions
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image-lat'
        fl.generate_flux_image = lambda *a, **kw: 'ok-image-text'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app
        self.assertEqual(app.st.session_state.prompt_image, 'ok-image-text')


if __name__ == '__main__':
    unittest.main()
