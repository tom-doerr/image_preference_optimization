import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestValueSidebar(unittest.TestCase):
    def test_predicted_values_present(self):
        st, writes = stub_with_writes()
        sys.modules['streamlit'] = st
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app  # noqa: F401

        text = "\n".join(writes)
        self.assertIn("V(left):", text)
        self.assertIn("V(right):", text)


if __name__ == '__main__':
    unittest.main()

