import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class TestIterEtaTiny(unittest.TestCase):
    def tearDown(self):
        for m in ("app", "streamlit", "flux_local"):
            sys.modules.pop(m, None)

    def test_iter_eta_accepts_twelve_decimals(self):
        st = types.ModuleType("streamlit")
        st.session_state = Session()
        st.set_page_config = lambda **_: None
        class _Exp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        st.sidebar = types.SimpleNamespace()
        st.title = lambda *_, **__: None
        st.caption = lambda *_, **__: None
        st.subheader = lambda *_, **__: None
        st.write = lambda *_, **__: None
        st.text_input = lambda *_, value="": value
        st.sidebar.text_input = staticmethod(lambda *_, value="": value)
        st.sidebar.subheader = staticmethod(lambda *_, **__: None)
        st.sidebar.header = staticmethod(lambda *_, **__: None)
        st.sidebar.expander = staticmethod(lambda *_, **__: _Exp())
        st.sidebar.selectbox = staticmethod(lambda label, options, **k: options[0])

        def num(label, *args, value=None, **kwargs):
            if label == "Iterative step (eta)":
                return 1e-12
            return value

        st.number_input = num
        st.sidebar.number_input = staticmethod(num)
        st.sidebar.write = lambda *a, **k: None
        st.slider = lambda *a, **k: 0
        st.sidebar.slider = staticmethod(lambda *a, **k: 0)
        st.button = lambda *a, **k: False
        st.sidebar.button = staticmethod(lambda *a, **k: False)
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        import app  # triggers controls

        self.assertIn("iter_eta", st.session_state)
        self.assertAlmostEqual(float(st.session_state["iter_eta"]), 1e-12, places=18)


if __name__ == "__main__":
    unittest.main()
