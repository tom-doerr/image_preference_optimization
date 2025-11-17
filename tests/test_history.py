import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class TestHistory(unittest.TestCase):
    def test_update_history_picks_best(self):
        st = types.ModuleType('streamlit')
        st.session_state = Session()
        st.set_page_config = lambda **_: None
        st.title = lambda *_, **__: None
        st.caption = lambda *_, **__: None
        st.subheader = lambda *_, **__: None
        st.text_input = lambda *_, value="": value
        st.number_input = lambda *_, value=None, **__: value
        st.slider = lambda *_, value=None, **__: value
        st.button = lambda *_, **__: False
        st.checkbox = lambda *_, **__: False
        st.image = lambda *_, **__: None
        class Sidebar:
            @staticmethod
            def selectbox(label, *args, **kwargs):
                return 'ridge' if 'Approach' in label else 'black-forest-labs/FLUX.1-schnell'
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
                return ''
        st.sidebar = Sidebar()
        class Col:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        st.columns = lambda n: (Col(), Col())
        st.write = lambda *_, **__: None
        st.experimental_rerun = lambda: None
        # Ensure fresh import for isolated state
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = st
        # Stub flux_local to allow autorun on import
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        import app
        # Initialize
        lstate = app.st.session_state.lstate
        # Favor +e0
        lstate.w[:] = 0
        lstate.w[0] = 1.0
        # Move mu strongly on e0 and update history
        lstate.mu[0] = 10.0
        app._update_history()
        self.assertEqual(app.st.session_state.mu_best_idx, len(app.st.session_state.mu_history) - 1)


if __name__ == '__main__':
    unittest.main()
