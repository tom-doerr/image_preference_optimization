import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_reset_only():
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value
    st.image = lambda *_, **__: None
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    def slider(label, *args, **kwargs):
        return kwargs.get('value', args[2] if len(args) >= 3 else 1.0)
    st.slider = slider
    def button(label, *a, **kw):
        return label == 'Reset'
    st.button = button
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
        def file_uploader(*_, **__): return None
        @staticmethod
        def button(*_, **__): return False
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
    return st


def stub_streamlit_revert_only():
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value
    st.image = lambda *_, **__: None
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    def slider(label, *args, **kwargs):
        return kwargs.get('value', args[2] if len(args) >= 3 else 1.0)
    st.slider = slider
    def button(label, *a, **kw):
        return label == 'Revert to Best μ'
    st.button = button
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
        def file_uploader(*_, **__): return None
        @staticmethod
        def button(*_, **__): return False
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
    return st


class TestResetThenRevert(unittest.TestCase):
    def test_reset_then_revert_best(self):
        # Pass 1: click Reset; persist state
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_reset_only()
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        import app
        self.assertEqual(app.st.session_state.lstate.step, 0)
        # After Reset we re-init state; images caches are cleared
        self.assertEqual(app.st.session_state.images, (None, None))

        # Revert UI removed; ensure no crash and μ preview not set
        del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_revert_only()
        sys.modules['flux_local'] = fl
        import app as app2
        self.assertTrue('mu_image' in app2.st.session_state and app2.st.session_state.mu_image is None)


if __name__ == '__main__':
    unittest.main()
