import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class TestIterEtaMorePrecision(unittest.TestCase):
    def tearDown(self):
        for m in ("app", "streamlit", "flux_local", "ui_controls"):
            sys.modules.pop(m, None)

    def test_iter_eta_accepts_four_decimals(self):
        # Stub Streamlit with a number_input returning a 4-decimal value
        st = types.ModuleType("streamlit")
        st.session_state = Session()
        st.set_page_config = lambda **_: None
        class _Exp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        st.sidebar = types.SimpleNamespace()
        # Basic text inputs to satisfy app import
        st.text_input = lambda *_, value="": value
        st.sidebar.text_input = staticmethod(lambda *_, value="": value)
        st.title = lambda *_, **__: None
        st.caption = lambda *_, **__: None
        st.subheader = lambda *_, **__: None
        st.write = lambda *_, **__: None
        st.sidebar.subheader = staticmethod(lambda *_, **__: None)
        st.sidebar.header = staticmethod(lambda *_, **__: None)
        st.sidebar.expander = staticmethod(lambda *_, **__: _Exp())
        st.sidebar.selectbox = staticmethod(lambda label, options, **k: options[0])

        def num(label, *args, value=None, **kwargs):
            if label == "Iterative step (eta)":
                return 0.0004
            return value

        st.number_input = num
        st.sidebar.number_input = staticmethod(num)
        st.sidebar.write = lambda *a, **k: None
        st.slider = lambda *a, **k: 0
        st.sidebar.slider = staticmethod(lambda *a, **k: 0)
        st.button = lambda *a, **k: False
        st.sidebar.button = staticmethod(lambda *a, **k: False)
        sys.modules["streamlit"] = st

        # Minimal stubs
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        # Import app to trigger controls
        import app  # noqa: F401

        # Confirm the eta value was set with 4-decimal precision
        self.assertIn("iter_eta", st.session_state)
        self.assertAlmostEqual(float(st.session_state["iter_eta"]), 0.0004, places=6)

        # build_pair_controls should return the updated eta
        from ui_controls import build_pair_controls

        _, _, _, _, _, _, eta = build_pair_controls(st, expanded=False)
        self.assertAlmostEqual(float(eta), 0.0004, places=6)


if __name__ == "__main__":
    unittest.main()
