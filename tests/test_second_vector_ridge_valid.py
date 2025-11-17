import sys
import types
import unittest
import numpy as np
from latent_opt import propose_latent_pair_ridge


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_defaults():
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value

    def slider(label, *args, **kwargs):
        # Return default values matching app: steps=20, guidance=3.5,
        # alpha=0.5, beta=0.5, trust_r=2.5
        if 'Alpha' in label:
            return 0.5
        if 'Beta' in label:
            return 0.5
        if 'Trust radius' in label:
            return 2.5
        return kwargs.get('value', args[2] if len(args) >= 3 else 1)

    st.slider = slider
    st.button = lambda *_, **__: False
    st.image = lambda *_, **__: None

    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return 'stabilityai/sd-turbo'
        header = staticmethod(lambda *a, **k: None)
        subheader = staticmethod(lambda *a, **k: None)
        download_button = staticmethod(lambda *a, **k: None)
        file_uploader = staticmethod(lambda *a, **k: None)
        text_input = staticmethod(lambda *a, **k: '')
        checkbox = staticmethod(lambda *a, **k: False)
        button = staticmethod(lambda *a, **k: False)
    st.sidebar = Sidebar()
    st.sidebar.write = lambda *a, **k: None

    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestSecondVectorRidgeValid(unittest.TestCase):
    def test_second_vector_matches_ridge_proposal(self):
        sys.modules['streamlit'] = stub_streamlit_defaults()
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.generate_flux_image = lambda *a, **kw: 'ok-image-text'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app  # noqa: F401
        st = sys.modules['streamlit']
        lstate = st.session_state.lstate
        _, zb = st.session_state.lz_pair
        # Expected ridge second vector with default alpha=0.5, beta=0.5, trust_r=2.5
        _, zb_expected = propose_latent_pair_ridge(lstate, alpha=0.5, beta=0.5, trust_r=2.5)
        self.assertTrue(np.allclose(zb, zb_expected))


if __name__ == '__main__':
    unittest.main()

