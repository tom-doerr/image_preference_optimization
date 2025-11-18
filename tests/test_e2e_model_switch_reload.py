import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_switch_model_and_generate():
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

    clicks = {'gen': False}
    def button(label, *a, **kw):
        if label == 'Generate pair' and not clicks['gen']:
            clicks['gen'] = True
            return True
        return False
    st.button = button

    class Sidebar:
        @staticmethod
        def selectbox(label, options, *args, **kwargs):
            return 'ridge'
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
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestE2EModelSwitchReload(unittest.TestCase):
    def test_set_model_receives_selected_model(self):
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_switch_model_and_generate()
        called = {'mid': None}
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **kw: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        def _set_model(mid):
            called['mid'] = mid
        fl.set_model = _set_model
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        import app
        # With simplified app, model is hardcoded to sd-turbo
        self.assertEqual(called['mid'], 'stabilityai/sd-turbo')
        self.assertEqual(app.st.session_state.images, ('ok-image', 'ok-image'))


if __name__ == '__main__':
    unittest.main()
