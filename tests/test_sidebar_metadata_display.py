import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes
import numpy as np
import io
import hashlib
import os


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def state_path_for_prompt(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:10]
    return f"latent_state_{h}.npz"


def stub_streamlit(prompt):
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
    st.text_input = lambda *_, value="": prompt
    st.button = lambda *_, **__: False
    writes = []
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
        def write(x): writes.append(str(x))
    st.sidebar = Sidebar()
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st, writes


class TestSidebarMetadataDisplay(unittest.TestCase):
    def test_metadata_lines_present_when_file_has_meta(self):
        prompt = 'meta prompt'
        path = state_path_for_prompt(prompt)
        # Create a valid state file then augment with metadata keys
        from latent_opt import init_latent_state, save_state
        st0 = init_latent_state(d=4, seed=0)
        save_state(st0, path)
        # Augment
        with np.load(path) as data:
            items = {k: data[k] for k in data.files}
        items['created_at'] = np.array('2025-11-13T00:00:00+00:00')
        items['app_version'] = np.array('0.1.0')
        buf = io.BytesIO()
        np.savez_compressed(buf, **items)
        with open(path, 'wb') as f:
            f.write(buf.getvalue())

        if 'app' in sys.modules:
            del sys.modules['app']
        st, writes = stub_with_writes()
        st.text_input = lambda *_, value="": prompt
        sys.modules['streamlit'] = st
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.generate_flux_image = lambda *a, **kw: 'ok-image-text'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl
        joined = '\n'.join(writes)
        self.assertIn('app_version:', joined)
        self.assertIn('created_at:', joined)

        # Cleanup
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == '__main__':
    unittest.main()
