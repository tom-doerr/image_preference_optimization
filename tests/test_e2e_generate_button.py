import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_click_generate_once():
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
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestE2EGenerateButton(unittest.TestCase):
    def test_generate_calls_set_model_and_images(self):
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_click_generate_once()
        calls = {'set_model': 0}
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        def _set_model(mid):
            calls['set_model'] += 1
        fl.set_model = _set_model
        sys.modules['flux_local'] = fl

        import app
        # After clicking generate, we expect images set and set_model invoked at least once
        self.assertEqual(app.st.session_state.images, ('ok-image', 'ok-image'))
        self.assertGreaterEqual(calls['set_model'], 1)


if __name__ == '__main__':
    unittest.main()

