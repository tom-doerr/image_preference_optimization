import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_old():
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value
    def _slider(*args, **kwargs):
        if len(args) >= 3:
            return args[2]
        return kwargs.get('value')
    st.slider = _slider
    st.button = lambda *_, **__: False
    st.image = lambda *_, **__: None
    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return 'ridge' if 'Approach' in label else 'stabilityai/sd-turbo'
        header = subheader = download_button = file_uploader = button = staticmethod(lambda *a, **k: None)
        text_input = staticmethod(lambda *a, **k: '')
        checkbox = staticmethod(lambda *a, **k: False)
        def write(*a, **k):
            return None
    st.sidebar = Sidebar()
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    # Only old API present
    st.experimental_rerun = lambda: None
    return st


def stub_streamlit_new():
    st = stub_streamlit_old()
    delattr(st, 'experimental_rerun')
    st.rerun = lambda: None
    return st


class TestRerunShim(unittest.TestCase):
    def tearDown(self):
        for m in ('app', 'flux_local', 'streamlit'):
            sys.modules.pop(m, None)

    def test_prefers_new_rerun(self):
        sys.modules['streamlit'] = stub_streamlit_new()
        # Stub flux_local to avoid CUDA
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        import app
        self.assertTrue(callable(app.st_rerun))

    def test_falls_back_to_experimental(self):
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_old()
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        import app
        self.assertTrue(callable(app.st_rerun))


if __name__ == '__main__':
    unittest.main()
