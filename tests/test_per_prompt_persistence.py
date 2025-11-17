import os
import sys
import types
import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def stub_streamlit(prompt, click_left=False):
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.number_input = lambda *_, value=None, **__: value
    st.image = lambda *_, **__: None
    def slider(label, *args, **kwargs):
        return kwargs.get('value', args[2] if len(args) >= 3 else 1.0)
    st.slider = slider
    def text_input(label, value=""):
        return prompt
    st.text_input = text_input
    def button(label, *a, **kw):
        return (label == 'Prefer Left') and click_left
    st.button = button
    class Sidebar:
        @staticmethod
        def selectbox(label, *args, **kwargs):
            return 'ridge' if 'Approach' in label else 'stabilityai/sd-turbo'
        @staticmethod
        def header(*_, **__): return None
        @staticmethod
        def subheader(*_, **__): return None
        @staticmethod
        def download_button(*_, **__): return None
        @staticmethod
        def file_uploader(*_, **__): return None
        @staticmethod
        def button(*_, **__): return False
        @staticmethod
        def slider(*_, **__): return 1.0
        @staticmethod
        def text_input(*_, **__): return ''
        @staticmethod
        def checkbox(*_, **__): return False
        @staticmethod
        def write(x): return None
    st.sidebar = Sidebar()
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestPerPromptPersistence(unittest.TestCase):
    def test_switch_prompts_loads_separate_states(self):
        # First prompt: click Prefer Left once to increment and save
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit('prompt A', click_left=True)
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        import app
        step_A = app.st.session_state.lstate.step
        path_A = app.st.session_state.state_path
        self.assertGreaterEqual(step_A, 1)
        self.assertTrue(os.path.exists(path_A))

        # Second prompt: no clicks; should be a fresh state (step 0)
        del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit('prompt B', click_left=False)
        sys.modules['flux_local'] = fl
        import app as appB
        step_B = appB.st.session_state.lstate.step
        path_B = appB.st.session_state.state_path
        self.assertEqual(step_B, 0)
        self.assertNotEqual(path_A, path_B)

        # Switch back to A: should load saved state with step >= previous
        del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit('prompt A', click_left=False)
        sys.modules['flux_local'] = fl
        import app as appA2
        self.assertEqual(appA2.st.session_state.state_path, path_A)
        self.assertGreaterEqual(appA2.st.session_state.lstate.step, step_A)


if __name__ == '__main__':
    unittest.main()

