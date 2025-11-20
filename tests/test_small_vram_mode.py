import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_small_vram():
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
            return 'black-forest-labs/FLUX.1-schnell'
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
        def checkbox(label, *_, **__):
            return True if '7 GB VRAM mode' in label else False
    st.sidebar = Sidebar()
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestSmallVramMode(unittest.TestCase):
    def test_small_vram_profile_removed(self):
        self.skipTest('7 GB VRAM mode removed from UI; adjust size/steps manually')


if __name__ == '__main__':
    unittest.main()
