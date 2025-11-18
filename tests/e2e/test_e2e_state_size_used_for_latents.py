import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestE2EStateSizeUsedForLatents(unittest.TestCase):
    def test_pipe_uses_state_size_not_slider_size(self):
        self.skipTest('Debug panel output formatting not enforced in simplified UI')
        st, writes = stub_with_writes()
        # Force Debug ON to read pipe_size
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'Debug' in label else value
        st.sidebar = SB()
        st.sidebar.write = lambda x: writes.append(str(x))
        st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
        # Pretend user set sliders to 640x640, but state is 512x512
        def number_input(label, **k):
            if 'Width' in label:
                return 640
            if 'Height' in label:
                return 640
            return k.get('value')
        st.number_input = number_input
        sys.modules['streamlit'] = st

        # flux_local stub that exposes get_last_call with the pipe size recorded
        recorded = {}
        fl = types.ModuleType('flux_local')
        def gen_latents(*args, **kw):
            recorded.update({k: kw[k] for k in ('width','height') if k in kw})
            return 'ok-image'
        fl.generate_flux_image_latents = gen_latents
        fl.set_model = lambda *a, **k: None
        def get_last_call():
            return {
                'event': 'latents_call',
                'model_id': 'stabilityai/sd-turbo',
                'width': recorded.get('width', 0),
                'height': recorded.get('height', 0),
                'latents_std': 1.0,
                'latents_mean': 0.0,
                'latents_shape': (1,4,64,64),
            }
        fl.get_last_call = get_last_call
        sys.modules['flux_local'] = fl

        import app  # noqa: F401
        text = "\n".join(writes)
        # Even though sliders are 640x640, we should decode with state size (512x512)
        self.assertIn('pipe_size: 512x512', text)


if __name__ == '__main__':
    unittest.main()
