import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class TestDefaultSteps(unittest.TestCase):
    def test_steps_slider_default_is_8(self):
        captured = {}

        def slider(label, min_value=None, max_value=None, value=None, step=None):
            if label == "Steps":
                captured['default'] = value
            return value

        st = types.ModuleType('streamlit')
        st.session_state = Session()
        st.set_page_config = lambda **_: None
        st.title = lambda *_, **__: None
        st.caption = lambda *_, **__: None
        st.subheader = lambda *_, **__: None
        st.text_input = lambda *_, value="": value
        st.number_input = lambda *_, value=None, **__: value
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
        class Col:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        st.columns = lambda n: (Col(), Col())
        st.write = lambda *_, **__: None
        st.experimental_rerun = lambda: None

        sys.modules['streamlit'] = st
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        sys.modules['flux_local'] = fl

        import app  # noqa: F401

        self.assertEqual(captured.get('default'), 8)


if __name__ == '__main__':
    unittest.main()

