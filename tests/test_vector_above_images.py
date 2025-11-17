import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_main_writes


class TestVectorAboveImages(unittest.TestCase):
    def test_vectors_render_above_pair_images(self):
        st, writes = stub_with_main_writes()
        sys.modules['streamlit'] = st
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app  # noqa: F401

        # Now we hide vector summaries; ensure they are not rendered
        rendered = "\n".join(writes)
        self.assertNotIn("z_a: d=", rendered)
        self.assertNotIn("z_b: d=", rendered)


if __name__ == '__main__':
    unittest.main()
