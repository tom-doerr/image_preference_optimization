import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class TestMuPreview(unittest.TestCase):
    def test_mu_image_set_on_generate_pair(self):
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
                return 'ridge' if 'Approach' in label else 'black-forest-labs/FLUX.1-schnell'
            @staticmethod
            def header(*_, **__):
                return None
            @staticmethod
            def checkbox(*_, **__):
                return False
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
        st.sidebar = Sidebar()
        Sidebar.slider = staticmethod(lambda *_, **__: 1.0)
        Sidebar.text_input = staticmethod(lambda *_, **__: '')
        class Col:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        st.columns = lambda n: (Col(), Col())
        st.write = lambda *_, **__: None
        st.experimental_rerun = lambda: None
        sys.modules['streamlit'] = st
        # Stub flux_local to allow autorun on import
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app
        app.generate_flux_image = lambda *a, **kw: 'ok-image'
        app.generate_pair()
        self.assertEqual(app.st.session_state.mu_image, 'ok-image')


if __name__ == '__main__':
    unittest.main()
