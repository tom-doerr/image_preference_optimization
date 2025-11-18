import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestDebugSidebar(unittest.TestCase):
    def test_debug_panel_renders_last_call(self):
        st, writes = stub_with_writes()
        # Force Debug checkbox ON
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'Debug' in label else value
        st.sidebar = SB()
        st.sidebar.write = lambda x: writes.append(str(x))
        st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
        sys.modules['streamlit'] = st

        # Stub flux_local
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        # Provide get_last_call with a fake payload
        def get_last_call():
            return {
                'event': 'latents_call',
                'model_id': 'stabilityai/sd-turbo',
                'latents_std': 0.99,
                'latents_mean': 0.01,
                'width': 512,
                'height': 512,
            }
        fl.get_last_call = get_last_call
        sys.modules['flux_local'] = fl

        import app  # noqa: F401
        text = "\n".join(writes)
        # Lines render via sidebar writes/metrics
        self.assertIn('model_id:', text)
        self.assertIn('latents_std:', text)
        self.assertIn('pipe_size:', text)


if __name__ == '__main__':
    unittest.main()
