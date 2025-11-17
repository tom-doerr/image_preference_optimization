import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def make_streamlit_with_prefer_left_once():
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value
    def slider(label, *args, **kwargs):
        if 'value' in kwargs and kwargs['value'] is not None:
            return kwargs['value']
        if len(args) >= 3:
            return args[2]
        if len(args) >= 1:
            return args[0]
        return 1.0
    st.slider = slider
    st.image = lambda *_, **__: None

    calls = {'prefer_left_clicked': False}
    def button(label, *a, **kw):
        if label == 'Prefer Left' and not calls['prefer_left_clicked']:
            calls['prefer_left_clicked'] = True
            return True
        return False
    st.button = button

    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return 'ridge' if 'Approach' in label else 'stabilityai/sd-turbo'
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


class TestE2EPreferButton(unittest.TestCase):
    def test_prefer_left_increments_step(self):
        # Ensure fresh import
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = make_streamlit_with_prefer_left_once()
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app
        # After import, our stub clicked Prefer Left once
        self.assertGreaterEqual(app.st.session_state.lstate.step, 1)


if __name__ == '__main__':
    unittest.main()
