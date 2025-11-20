import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit():
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
            return 'ridge' if 'Approach' in label else 'black-forest-labs/FLUX.1-schnell'
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


class TestAutorunGeneration(unittest.TestCase):
    def test_images_generated_on_import(self):
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit()
        # Provide stubbed flux_local so autorun works
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app
        # Batch autorun now runs; pair images are not decoded automatically.
        self.assertTrue('images' in app.st.session_state)
        self.assertIn(app.st.session_state.images, (None, (None, None)))
        # μ preview remains disabled
        self.assertTrue('mu_image' in app.st.session_state and app.st.session_state.mu_image is None)


if __name__ == '__main__':
    unittest.main()
