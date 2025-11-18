import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestGenLoopHelpers(unittest.TestCase):
    def test_decode_records_stats_for_both_sides(self):
        # Minimal stub streamlit; no slots so rendering is skipped
        st = stub_basic(pre_images=False)
        sys.modules['streamlit'] = st

        # Stub flux_local to simulate two consecutive calls
        calls = {'n': -1}
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **k: 'ok-text'
        def gen_latents(*args, **kw):
            calls['n'] += 1
            return 'ok-image'
        def get_last_call():
            return {'event': 'latents_call', 'call_index': calls['n']}
        fl.generate_flux_image_latents = gen_latents
        fl.get_last_call = get_last_call
        fl.set_model = lambda *a, **k: None
        sys.modules['flux_local'] = fl

        import app  # autorun will call generate_pair once

        stats = app.st.session_state.get('img_stats') or {}
        self.assertIn('left', stats)
        self.assertIn('right', stats)
        self.assertIsInstance(stats['left'].get('call_index'), int)
        self.assertIsInstance(stats['right'].get('call_index'), int)


if __name__ == '__main__':
    unittest.main()
