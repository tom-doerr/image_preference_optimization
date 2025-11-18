import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestE2EDebugWarnOnLowStd(unittest.TestCase):
    def test_warns_when_latents_std_is_zeroish(self):
        st, writes = stub_with_writes()
        # Force Debug ON and make width/height differ from state to show sizes line
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'Debug' in label else value
        st.sidebar = SB()
        st.sidebar.write = lambda x: writes.append(str(x))
        st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
        # Make sliders return something different to exercise size note
        st.number_input = lambda label, **k: 640 if 'Width' in label else 640
        sys.modules['streamlit'] = st

        # Stub flux_local to emit a last_call with std≈0
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **kw: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        def get_last_call():
            return {
                'event': 'latents_call',
                'model_id': 'stabilityai/sd-turbo',
                'latents_std': 0.0,
                'latents_mean': 0.0,
                'width': 512,
                'height': 512,
                'latents_shape': (1,4,64,64),
            }
        fl.get_last_call = get_last_call
        sys.modules['flux_local'] = fl

        import app  # noqa: F401
        text = "\n".join(writes)
        self.assertIn('warn: latents std', text)


if __name__ == '__main__':
    unittest.main()
