import sys
import types
import unittest
import hashlib


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def hash10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def stub_streamlit_with_captions(prompts):
    st = types.ModuleType("streamlit")
    st.session_state = Session()
    st.session_state.recent_prompts = prompts
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *args, **kwargs: captions.append(str(args[0]) if args else "")
    st.subheader = lambda *_, **__: None
    st.number_input = lambda *_, value=None, **__: value
    st.image = lambda *_, **__: None

    def slider(label, *args, **kwargs):
        return kwargs.get("value", args[2] if len(args) >= 3 else 1.0)

    st.slider = slider
    st.text_input = lambda *_, value="": prompts[0]
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


class TestFooterRecentPrompts(unittest.TestCase):
    def test_footer_shows_recent_hashes(self):
        global captions
        captions = []
        prompts = [
            "a very long prompt text that will be truncated here",
            "second prompt",
            "third prompt",
        ]
        if "app" in sys.modules:
            del sys.modules["app"]
        sys.modules["streamlit"] = stub_streamlit_with_captions(prompts)
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        joined = "\n".join(captions)
        self.assertIn("Recent states:", joined)
        self.assertIn(hash10(prompts[0]), joined)
        self.assertIn(hash10(prompts[1]), joined)


if __name__ == "__main__":
    unittest.main()
