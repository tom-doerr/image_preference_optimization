import unittest
import types
import sys


class DummyExecutor:
    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.submitted = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    class _Fut:
        def __init__(self, res):
            self._res = res

        def result(self):
            return self._res

    def submit(self, fn, *a, **kw):
        self.submitted.append((fn, a, kw))
        return DummyExecutor._Fut(fn(*a, **kw))


class TestParallelGen(unittest.TestCase):
    def test_executor_used_for_pair(self):
        # Stub streamlit to import app
        st = types.ModuleType("streamlit")

        class Session(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__

        st.session_state = Session()
        st.set_page_config = lambda **_: None
        st.title = lambda *_, **__: None
        st.caption = lambda *_, **__: None
        st.text_input = lambda *_, value="": value
        st.number_input = lambda *_, value=None, **__: value
        st.slider = lambda *_, value=None, **__: value
        st.checkbox = lambda *_, **__: False

        class Sidebar:
            @staticmethod
            def selectbox(label, *args, **kwargs):
                return (
                    "ridge"
                    if "Approach" in label
                    else "black-forest-labs/FLUX.1-schnell"
                )

            @staticmethod
            def header(*_, **__):
                return None

            @staticmethod
            def checkbox(*_, **__):
                return False

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

        st.sidebar = Sidebar()
        st.button = lambda *_, **__: False
        st.image = lambda *_, **__: None

        class Col:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        st.columns = lambda n: (Col(), Col())
        st.write = lambda *_, **__: None
        st.experimental_rerun = lambda: None
        sys.modules["streamlit"] = st
        # Stub flux_local to allow autorun on import
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        import app

        # Monkey-patch executor and generator
        app.futures.ThreadPoolExecutor = lambda max_workers=2: DummyExecutor(
            max_workers=max_workers
        )
        app.generate_flux_image = lambda *a, **kw: "ok-image"

        # Ensure state exists and call
        app.generate_pair()
        self.assertEqual(app.st.session_state.images, ("ok-image", "ok-image"))


if __name__ == "__main__":
    unittest.main()
