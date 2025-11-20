import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit_capture_sliders():
    st = types.ModuleType("streamlit")
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value

    calls = []

    def slider(label, *args, **kwargs):
        # capture label, help text, and default value
        if len(args) >= 3:
            default_val = args[2]
        else:
            default_val = kwargs.get("value")
        calls.append({"label": label, "help": kwargs.get("help"), "value": default_val})
        # return the default value
        return default_val

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
    return st, calls


class TestSliderHelp(unittest.TestCase):
    def test_alpha_beta_have_tooltips(self):
        st, calls = stub_streamlit_capture_sliders()
        # Call build_pair_controls directly to capture slider labels/help.
        from ui_controls import build_pair_controls

        build_pair_controls(st, expanded=False)
        labels = {c["label"]: c.get("help") for c in calls if c.get("label")}
        self.assertIn("Alpha (ridge d1)", labels)
        self.assertIn("Beta (ridge d2)", labels)
        self.assertIsNotNone(labels["Alpha (ridge d1)"])
        self.assertIsNotNone(labels["Beta (ridge d2)"])
        self.assertIn("d1", labels["Alpha (ridge d1)"])
        self.assertIn("d2", labels["Beta (ridge d2)"])

    def test_iter_eta_default_positive(self):
        st, _ = stub_streamlit_capture_sliders()
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        # Numeric eta input in app.py should set a positive default.
        self.assertIn("iter_eta", st.session_state)
        self.assertGreater(float(st.session_state["iter_eta"]), 0.0)

    def test_iter_eta_numeric_updates_state(self):
        st, _ = stub_streamlit_capture_sliders()

        # Numeric input: override only the eta field to simulate a user edit.
        def num(label, *args, value=None, **kwargs):
            if label == "Iterative step (eta)":
                return 0.3
            return value

        st.number_input = num
        st.sidebar.number_input = staticmethod(num)

        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl

        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        # Numeric eta input should have updated the shared state.
        self.assertIn("iter_eta", st.session_state)
        self.assertAlmostEqual(float(st.session_state["iter_eta"]), 0.3, places=6)

        # build_pair_controls should now see and return the updated eta value.
        from ui_controls import build_pair_controls

        alpha, beta, trust_r, lr_mu_ui, gamma_orth, iter_steps, iter_eta = (
            build_pair_controls(st, expanded=False)
        )
        self.assertAlmostEqual(float(iter_eta), 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
