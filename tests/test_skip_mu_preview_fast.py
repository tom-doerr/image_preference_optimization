import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_mu_off():
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
    st.image = lambda *_, **__: None
    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return 'stabilityai/sd-turbo'
        header = staticmethod(lambda *a, **k: None)
        subheader = staticmethod(lambda *a, **k: None)
        download_button = staticmethod(lambda *a, **k: None)
        file_uploader = staticmethod(lambda *a, **k: None)
        text_input = staticmethod(lambda *a, **k: '')
        checkbox = staticmethod(lambda label, *a, **k: False if 'μ preview' in label else False)
        write = staticmethod(lambda *a, **k: None)
        button = staticmethod(lambda *a, **k: False)
    st.sidebar = Sidebar()
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestSkipMuPreviewFast(unittest.TestCase):
    def test_mu_preview_skipped_when_off(self):
        sys.modules['streamlit'] = stub_streamlit_mu_off()
        # Stub flux_local so autorun on import doesn't require GPU/network
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.generate_flux_image = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        import app
        # After autorun, μ image should be skipped when preview is off
        self.assertIsNone(app.st.session_state.mu_image)
        self.assertEqual(app.st.session_state.images, ('ok-image', 'ok-image'))


if __name__ == '__main__':
    unittest.main()
