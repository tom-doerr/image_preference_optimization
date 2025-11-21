import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit():
    st = types.ModuleType("streamlit")
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value
    st.slider = lambda *_, value=None, **__: value
    st.button = lambda *_, **__: False
    st.image = lambda *_, **__: None

    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return "ridge" if "Approach" in label else "stabilityai/sd-turbo"

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
            return ""

        @staticmethod
        def checkbox(*_, **__):
            return False

    st.sidebar = Sidebar()

    class Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestSetModelCalledOnAutorun(unittest.TestCase):
    def test_set_model_invoked(self):
        if "app" in sys.modules:
            del sys.modules["app"]
        sys.modules["streamlit"] = stub_streamlit()
        calls = {"n": 0}
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"

        def set_model(mid):
            calls["n"] += 1

        fl.set_model = set_model
        sys.modules["flux_local"] = fl
        import app  # trigger import-time set_model(DEFAULT_MODEL)
        self.assertGreaterEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
