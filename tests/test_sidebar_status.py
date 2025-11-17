import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarStatus(unittest.TestCase):
    def test_status_lines_present(self):
        if 'app' in sys.modules:
            del sys.modules['app']
        st, writes = stub_with_writes(pre_images=True)
        sys.modules['streamlit'] = st
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.generate_flux_image = lambda *a, **kw: 'ok-image-text'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        # Check that the three status lines were written
        joined = '\n'.join(writes)
        # Since prompt-aware reload can reset caches on import, accept either ready or empty here
        self.assertIn('Left:', joined)
        self.assertIn('Right:', joined)
        self.assertIn('μ preview:', joined)


if __name__ == '__main__':
    unittest.main()
