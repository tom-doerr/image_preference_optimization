import sys
import types
import unittest


def stub_streamlit_capture():
    from tests.helpers.st_streamlit import stub_basic

    st = stub_basic(pre_images=False)
    return st, st.sidebar_writes


class TestSidebarLatentDim(unittest.TestCase):
    def test_latent_dim_line_present(self):
        st, writes = stub_streamlit_capture()
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.generate_flux_image = lambda *a, **kw: "ok-image-text"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl
        import app  # noqa: F401

        text = "\n".join(writes)
        self.assertIn("Latent dim:", text)


if __name__ == "__main__":
    unittest.main()
