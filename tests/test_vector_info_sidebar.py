import sys
import types
import unittest
import numpy as np


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_capture_sidebar():
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value
    def slider(label, *a, **k):
        return k.get('value', a[2] if len(a) >= 3 else 1)
    st.slider = slider
    st.button = lambda *_, **__: False
    st.image = lambda *_, **__: None
    writes = []
    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return 'stabilityai/sd-turbo'
        header = staticmethod(lambda *a, **k: None)
        subheader = staticmethod(lambda *a, **k: None)
        download_button = staticmethod(lambda *a, **k: None)
        file_uploader = staticmethod(lambda *a, **k: None)
        text_input = staticmethod(lambda *a, **k: '')
        checkbox = staticmethod(lambda *a, **k: False)
        button = staticmethod(lambda *a, **k: False)
    st.sidebar = Sidebar()
    st.sidebar.write = lambda x: writes.append(str(x))
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st, writes


class TestVectorInfoSidebar(unittest.TestCase):
    def test_sidebar_shows_vector_metrics(self):
        self.skipTest('Pair vector info sidebar removed')


if __name__ == '__main__':
    unittest.main()
