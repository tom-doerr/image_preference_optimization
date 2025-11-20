import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class DummyUpload:
    def __init__(self, data):
        self._data = data
    def read(self):
        return self._data


def stub_streamlit_import_with_upload(data_bytes):
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
    st.button = lambda *_, **__: False

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
        def file_uploader(*_, **__):
            return DummyUpload(data_bytes)
        @staticmethod
        def button(label, *_, **__):
            return label == 'Load uploaded state'
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


class TestImportStateMuHistory(unittest.TestCase):
    def test_mu_history_populates_from_import(self):
        self.skipTest('Upload/import UI removed from app; export only')
        return


if __name__ == '__main__':
    unittest.main()
