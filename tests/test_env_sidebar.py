import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class Recorder:
    def __init__(self):
        self.lines = []
    def write(self, x):
        self.lines.append(str(x))


def stub_streamlit(rec):
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
        selectbox = staticmethod(lambda label, *a, **k: 'ridge' if 'Approach' in label else 'stabilityai/sd-turbo')
        header = staticmethod(lambda *a, **k: None)
        subheader = staticmethod(lambda *a, **k: None)
        download_button = staticmethod(lambda *a, **k: None)
        file_uploader = staticmethod(lambda *a, **k: None)
        text_input = staticmethod(lambda *a, **k: '')
        checkbox = staticmethod(lambda *a, **k: False)
        button = staticmethod(lambda *a, **k: False)
    st.sidebar = Sidebar()
    # Attach write at instance level so app can call it
    st.sidebar.write = lambda x: rec.write(x)
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestEnvSidebar(unittest.TestCase):
    def tearDown(self):
        for m in ('app', 'flux_local', 'streamlit'):
            sys.modules.pop(m, None)

    def test_env_info_helper_has_keys(self):
        from env_info import get_env_summary
        info = get_env_summary()
        self.assertIn('python', info)
        self.assertIn('torch', info)


if __name__ == '__main__':
    unittest.main()
