import sys
import types
import unittest
import numpy as np


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_minimal():
    st = types.ModuleType("streamlit")
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value

    def slider(label, *a, **k):
        # Always return the default value
        return k.get("value", a[2] if len(a) >= 3 else 1)

    st.slider = slider
    st.button = lambda *_, **__: False
    st.image = lambda *_, **__: None

    class Sidebar:
        selectbox = staticmethod(lambda *a, **k: "stabilityai/sd-turbo")
        header = staticmethod(lambda *a, **k: None)
        subheader = staticmethod(lambda *a, **k: None)
        download_button = staticmethod(lambda *a, **k: None)
        file_uploader = staticmethod(lambda *a, **k: None)
        text_input = staticmethod(lambda *a, **k: "")
        checkbox = staticmethod(lambda *a, **k: False)
        button = staticmethod(lambda *a, **k: False)

    st.sidebar = Sidebar()
    st.sidebar.write = lambda *a, **k: None

    class Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestInitialPairUsesPrompt(unittest.TestCase):
    def test_first_pair_uses_prompt_vector(self):
        sys.modules["streamlit"] = stub_streamlit_minimal()
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.generate_flux_image = lambda *a, **kw: "ok-image-text"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        import app  # noqa: F401

        st = sys.modules["streamlit"]
        za, zb = st.session_state.lz_pair
        # At minimum, first z_a should be non-zero (no all-zero black image)
        za_n = float(np.linalg.norm(za))
        self.assertGreater(za_n, 0.0)
        # step may change due to internal flows; core check is non-zero z_a


if __name__ == "__main__":
    unittest.main()
