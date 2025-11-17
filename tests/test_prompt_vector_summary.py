import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_main_writes


class TestPromptVectorSummary(unittest.TestCase):
    def test_prompt_vector_is_rendered(self):
        st, writes = stub_with_main_writes()
        sys.modules['streamlit'] = st
        fl = types.ModuleType('flux_local')
        # Only provide latents path; app should still generate prompt vector + image
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app  # noqa: F401

        rendered = "\n".join(writes)
        self.assertNotIn("z_prompt: d=", rendered)


if __name__ == '__main__':
    unittest.main()
