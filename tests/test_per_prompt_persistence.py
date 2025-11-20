import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit(prompt, click_left=False):
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

    def text_input(label, value=""):
        return prompt

    st.text_input = text_input

    def button(label, *a, **kw):
        return (label == "Prefer Left") and click_left

    st.button = button

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


class TestPerPromptPersistence(unittest.TestCase):
    def test_pair_mode_removed(self):
        self.skipTest(
            "Pair mode removed; per-prompt persistence is covered via dataset tests"
        )


if __name__ == "__main__":
    unittest.main()
