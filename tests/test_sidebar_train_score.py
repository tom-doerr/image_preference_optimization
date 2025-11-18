import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarTrainScore(unittest.TestCase):
    def test_shows_train_score_metric(self):
        st, writes = stub_with_writes()
        sys.modules['streamlit'] = st
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **kw: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        import app  # noqa: F401

        out = "\n".join(writes)
        self.assertIn('Train score:', out)


if __name__ == '__main__':
    unittest.main()

