import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_capture_sliders():
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value

    calls = []
    def slider(label, *args, **kwargs):
        calls.append({"label": label, "help": kwargs.get("help")})
        # return the default value
        if len(args) >= 3:
            return args[2]
        return kwargs.get('value')
    st.slider = slider
    st.button = lambda *_, **__: False
    st.image = lambda *_, **__: None
    class Sidebar:
        selectbox = staticmethod(lambda *a, **k: 'stabilityai/sd-turbo')
        header = staticmethod(lambda *a, **k: None)
        subheader = staticmethod(lambda *a, **k: None)
        download_button = staticmethod(lambda *a, **k: None)
        file_uploader = staticmethod(lambda *a, **k: None)
        text_input = staticmethod(lambda *a, **k: '')
        checkbox = staticmethod(lambda *a, **k: False)
        button = staticmethod(lambda *a, **k: False)
    st.sidebar = Sidebar()
    st.sidebar.write = lambda *a, **k: None
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st, calls


class TestSliderHelp(unittest.TestCase):
    def test_alpha_beta_have_tooltips(self):
        st, calls = stub_streamlit_capture_sliders()
        sys.modules['streamlit'] = st
        # Stub flux_local to avoid GPU work
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app  # noqa: F401
        labels = {c['label']: c.get('help') for c in calls if c.get('label')}
        self.assertIn('Alpha (ridge d1)', labels)
        self.assertIn('Beta (ridge d2)', labels)
        self.assertIsNotNone(labels['Alpha (ridge d1)'])
        self.assertIsNotNone(labels['Beta (ridge d2)'])
        self.assertIn('d1', labels['Alpha (ridge d1)'])
        self.assertIn('d2', labels['Beta (ridge d2)'])


if __name__ == '__main__':
    unittest.main()
