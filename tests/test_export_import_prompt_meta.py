import sys
import types
import unittest
import numpy as np
import io


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class DummyUpload:
    def __init__(self, data):
        self._data = data
    def read(self):
        return self._data


def stub_streamlit_export(prompt):
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
    st.button = lambda *_, **__: False
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
        @staticmethod
        def warning(x): return None
    st.sidebar = Sidebar()
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


def stub_streamlit_import_with_mismatch(current_prompt, uploaded_bytes, warnings):
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
        return current_prompt
    st.text_input = text_input
    st.button = lambda *a, **k: False
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
        def file_uploader(*_, **__):
            return DummyUpload(uploaded_bytes)
        @staticmethod
        def button(label, *_, **__):
            return label == 'Load uploaded state'
        @staticmethod
        def slider(*_, **__): return 1.0
        @staticmethod
        def text_input(*_, **__): return ''
        @staticmethod
        def checkbox(*_, **__): return False
        @staticmethod
        def write(x): return None
        @staticmethod
        def warning(x): warnings.append(str(x))
    st.sidebar = Sidebar()
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestExportImportPromptMeta(unittest.TestCase):
    def test_export_includes_prompt(self):
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_export('prompt A')
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        import app
        data = app._export_state_bytes(app.st.session_state.lstate, app.st.session_state.prompt)
        arr = np.load(io.BytesIO(data))
        self.assertIn('prompt', arr.files)
        self.assertEqual(arr['prompt'].item(), 'prompt A')

    def test_import_mismatch_prompts_warns(self):
        # Build exported bytes for prompt B
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_export('prompt B')
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        import app as maker
        data = maker._export_state_bytes(maker.st.session_state.lstate, 'prompt B')

        # Now try to import into prompt A and capture warning
        warnings = []
        del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_import_with_mismatch('prompt A', data, warnings)
        sys.modules['flux_local'] = fl
        import app  # trigger import-time upload handling and warning
        self.assertTrue(any('different prompt' in w for w in warnings))

    def test_import_mismatch_switch_button_loads(self):
        # Build exported bytes for prompt B
        if 'app' in sys.modules:
            del sys.modules['app']
        sys.modules['streamlit'] = stub_streamlit_export('prompt B')
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        import app as maker
        data = maker._export_state_bytes(maker.st.session_state.lstate, 'prompt B')

        # Now try to import into prompt A and click the switch button
        warnings = []
        del sys.modules['app']
        def stub_streamlit_import_with_switch():
            st = stub_streamlit_import_with_mismatch('prompt A', data, warnings)
            # Monkey-patch sidebar.button to return True for our new switch action
            orig_btn = st.sidebar.button
            def btn(label, *a, **k):
                if label == 'Switch to uploaded prompt and load now':
                    return True
                return orig_btn(label, *a, **k)
            st.sidebar.button = btn
            return st
        sys.modules['streamlit'] = stub_streamlit_import_with_switch()
        sys.modules['flux_local'] = fl
        import app as consumer2
        # After switch, prompt should match uploaded prompt
        self.assertEqual(consumer2.st.session_state.prompt, 'prompt B')


if __name__ == '__main__':
    unittest.main()
