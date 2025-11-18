import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestPrefetchNextPair(unittest.TestCase):
    def test_prefetch_object_created_after_generate(self):
        st = stub_basic(pre_images=False)
        # Provide a slider on sidebar in this stub
        class SB(st.sidebar.__class__):
            @staticmethod
            def slider(label, *a, **k):
                return k.get('value', a[2] if len(a) >= 3 else 1)
        st.sidebar = SB()
        sys.modules['streamlit'] = st
        # Stub flux_local to be fast
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **k: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        import app
        # Explicitly schedule a prefetch, then assert presence
        app._prefetch_next_for_generate()
        npf = app.st.session_state.get('next_prefetch')
        self.assertIsNotNone(npf)
        self.assertIn('za', npf)
        self.assertIn('zb', npf)


if __name__ == '__main__':
    unittest.main()
