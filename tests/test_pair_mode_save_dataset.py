import os
import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic
from persistence import dataset_rows_for_prompt, dataset_path_for_prompt


class TestPairModeSaveDataset(unittest.TestCase):
    def test_prefer_left_appends_two_rows(self):
        st = stub_basic(pre_images=False)
        prompt = 'pair mode save dataset test'
        st.text_input = lambda *_, value="": prompt
        # Sidebar stubs: select Pair mode
        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, options, *a, **k):
                return 'Pair (A/B)'
            @staticmethod
            def expander(*a, **k):
                class _E:
                    def __enter__(self): return self
                    def __exit__(self, *e): return False
                return _E()
            @staticmethod
            def slider(*a, **k):
                return k.get('value', 1)
        st.sidebar = SB()
        sys.modules['streamlit'] = st

        # Fast generator stubs
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **k: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        # Ensure clean dataset file
        path = dataset_path_for_prompt(prompt)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

        import app
        # Simulate prefer left → +1 for z_a and -1 for z_b
        app._choose_preference('a')
        self.assertGreaterEqual(dataset_rows_for_prompt(app.base_prompt), 2)


if __name__ == '__main__':
    unittest.main()

