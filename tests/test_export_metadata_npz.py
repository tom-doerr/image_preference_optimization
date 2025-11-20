import sys
import types
import unittest
import numpy as np
import io


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit(prompt="p"):
    st = types.ModuleType("streamlit")
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.number_input = lambda *_, value=None, **__: value
    st.image = lambda *_, **__: None

    def slider(label, *args, **kwargs):
        return kwargs.get("value", args[2] if len(args) >= 3 else 1.0)

    st.slider = slider
    st.text_input = lambda *_, value="": prompt
    st.button = lambda *_, **__: False

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

        @staticmethod
        def write(x):
            return None

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


class TestExportMetadataNPZ(unittest.TestCase):
    def test_export_includes_created_and_version(self):
        if "app" in sys.modules:
            del sys.modules["app"]
        sys.modules["streamlit"] = stub_streamlit("prompt meta")
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl
        import app

        data = app._export_state_bytes(
            app.st.session_state.lstate, app.st.session_state.prompt
        )
        arr = np.load(io.BytesIO(data))
        self.assertIn("created_at", arr.files)
        self.assertIn("app_version", arr.files)
        self.assertIsInstance(arr["created_at"].item(), str)
        self.assertRegex(arr["created_at"].item(), "T")


if __name__ == "__main__":
    unittest.main()
