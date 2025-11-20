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


class TestE2EExportImportCycle(unittest.TestCase):
    def test_export_import_roundtrip(self):
        sys.modules["streamlit"] = stub_streamlit()
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        import app

        # Do an update to change state
        z_a, z_b = app.st.session_state.lz_pair
        app.update_latent_ridge(
            app.st.session_state.lstate, z_a, z_b, "a", feats_a=z_a, feats_b=z_b
        )
        # Export via app namespace
        data = app.dumps_state(app.st.session_state.lstate)
        # Import into a new state
        st2 = app.loads_state(data)
        # Validate important fields survived
        self.assertEqual(st2.d, app.st.session_state.lstate.d)
        self.assertEqual(st2.step, app.st.session_state.lstate.step)
        self.assertEqual(st2.mu.shape, app.st.session_state.lstate.mu.shape)
        self.assertEqual(st2.w.shape, app.st.session_state.lstate.w.shape)


if __name__ == "__main__":
    unittest.main()
