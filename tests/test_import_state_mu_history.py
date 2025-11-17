import sys
import types
import unittest
import numpy as np


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class DummyUpload:
    def __init__(self, data):
        self._data = data
    def read(self):
        return self._data


def stub_streamlit_import_with_upload(data_bytes):
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value
    st.image = lambda *_, **__: None
    def slider(label, *args, **kwargs):
        return kwargs.get('value', args[2] if len(args) >= 3 else 1.0)
    st.slider = slider
    st.button = lambda *_, **__: False

    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return 'ridge' if 'Approach' in label else 'stabilityai/sd-turbo'
        @staticmethod
        def header(*_, **__): return None
        @staticmethod
        def subheader(*_, **__): return None
        @staticmethod
        def download_button(*_, **__): return None
        @staticmethod
        def file_uploader(*_, **__):
            return DummyUpload(data_bytes)
        @staticmethod
        def button(label, *_, **__):
            return label == 'Load uploaded state'
        @staticmethod
        def slider(*_, **__): return 1.0
        @staticmethod
        def text_input(*_, **__): return ''
        @staticmethod
        def checkbox(*_, **__): return False
    st.sidebar = Sidebar()

    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestImportStateMuHistory(unittest.TestCase):
    def test_mu_history_populates_from_import(self):
        # Build a state with custom mu_hist and export bytes
        from latent_opt import init_latent_state, dumps_state
        st0 = init_latent_state(seed=0)
        st0.mu_hist = np.stack([np.zeros(4), np.ones(4)], axis=0)
        data = dumps_state(st0)

        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_import_with_upload(data)
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app
        self.assertGreaterEqual(len(app.st.session_state.mu_history), 2)


if __name__ == '__main__':
    unittest.main()
