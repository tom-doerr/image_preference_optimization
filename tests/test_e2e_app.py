import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def make_stub_streamlit():
    st = types.ModuleType("streamlit")
    st.session_state = Session()

    def _ret(x=None, **kwargs):
        return x

    def text_input(label, value=""):
        return value

    def number_input(
        label, min_value=None, max_value=None, step=None, value=None, **kwargs
    ):
        return value

    def slider(label, min_value=None, max_value=None, value=None, step=None):
        return value

    def button(*args, **kwargs):
        return False

    def image(*args, **kwargs):
        return None

    class Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def columns(n):
        return (Col(), Col())

    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = text_input
    st.number_input = number_input
    st.slider = slider

    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return (
                "ridge" if "Approach" in label else "black-forest-labs/FLUX.1-schnell"
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
    st.button = button
    st.checkbox = lambda *_, **__: False
    st.image = image
    st.columns = columns
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestAppE2E(unittest.TestCase):
    def test_end_to_end_round(self):
        sys.modules["streamlit"] = make_stub_streamlit()
        # Stub flux_local so autorun on import doesn't require GPU/network
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl
        import os
        os.environ["IPO_AUTORUN"] = "1"
        import app  # type: ignore

        # Mock image generator to avoid network/GPU
        app.generate_flux_image = lambda prompt, **k: "ok-image"

        # Ensure initial latent state exists
        self.assertIn("lstate", app.st.session_state)
        self.assertIn("lz_pair", app.st.session_state)

        # Capture before
        before_step = app.st.session_state.lstate.step

        # Generate pair and simulate a user choice
        app.generate_pair()
        url_a, url_b = app.st.session_state.images
        self.assertEqual(url_a, "ok-image")
        self.assertEqual(url_b, "ok-image")

        z_a, z_b = app.st.session_state.lz_pair
        # Choose 'a'
        app.update_latent_ridge(app.st.session_state.lstate, z_a, z_b, "a")

        after_step = app.st.session_state.lstate.step
        self.assertEqual(after_step, before_step + 1)


if __name__ == "__main__":
    unittest.main()
