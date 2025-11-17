import sys
import types
import unittest
import numpy as np


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit(approach='ridge'):
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value
    st.slider = lambda *_, value=None, **__: value
    st.button = lambda *_, **__: False
    st.checkbox = lambda *_, **__: False
    st.image = lambda *_, **__: None
    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return approach if 'Approach' in label else 'black-forest-labs/FLUX.1-schnell'
        @staticmethod
        def header(*_, **__):
            return None
        @staticmethod
        def subheader(*_, **__):
            return None
        @staticmethod
        def download_button(*_, **__):
            return None
        @staticmethod
        def file_uploader(*_, **__):
            return None
        @staticmethod
        def button(*_, **__):
            return False
        @staticmethod
        def slider(*_, **__):
            return 1.0
        @staticmethod
        def text_input(*_, **__):
            return ''
        @staticmethod
        def checkbox(*_, **__):
            return False
    st.sidebar = Sidebar()
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestE2ERidgeFlow(unittest.TestCase):
    def test_ridge_end_to_end(self):
        sys.modules['streamlit'] = stub_streamlit('ridge')
        # Stub flux_local to allow autorun on import
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        import app

        # Stub heavy calls
        calls = {}
        app.set_model = lambda mid: calls.setdefault('model', mid)
        app.generate_flux_image = lambda *a, **kw: 'ok-image'

        # Generate pair (stubbed)
        app.generate_pair()
        self.assertEqual(app.st.session_state.images, ('ok-image', 'ok-image'))

        # Simulate a ridge update with deterministic latent features
        d = app.st.session_state.lstate.d
        import numpy as _np
        z_a = _np.ones(d)
        z_b = -_np.ones(d)
        before_step = app.st.session_state.lstate.step
        app.update_latent_ridge(app.st.session_state.lstate, z_a, z_b, 'a', feats_a=z_a, feats_b=z_b)
        self.assertEqual(app.st.session_state.lstate.step, before_step + 1)
        # w should have non-zero norm after update
        self.assertGreater(float(np.linalg.norm(app.st.session_state.lstate.w)), 0.0)


if __name__ == '__main__':
    unittest.main()
