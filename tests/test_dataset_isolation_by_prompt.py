import os
import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic
from persistence import dataset_rows_for_prompt, dataset_path_for_prompt


class TestDatasetIsolation(unittest.TestCase):
    def test_datasets_are_separate_per_prompt(self):
        st = stub_basic(pre_images=False)
        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, options, *a, **k): return 'Batch curation'
            @staticmethod
            def expander(*a, **k):
                class _E:
                    def __enter__(self): return self
                    def __exit__(self, *e): return False
                return _E()
            @staticmethod
            def slider(*a, **k): return k.get('value', 1)
            @staticmethod
            def checkbox(*a, **k): return False
        st.sidebar = SB()
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **k: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        # Prompt A
        prompt_a = 'dataset iso A'
        st.text_input = lambda *_, value="": prompt_a
        # Clean A
        try:
            os.remove(dataset_path_for_prompt(prompt_a))
        except FileNotFoundError:
            pass
        import app
        app._curation_add(1, app.st.session_state.cur_batch[0])
        rows_a = dataset_rows_for_prompt(app.base_prompt)
        self.assertGreaterEqual(rows_a, 1)

        # Prompt B
        del sys.modules['app']
        prompt_b = 'dataset iso B'
        st.text_input = lambda *_, value="": prompt_b
        try:
            os.remove(dataset_path_for_prompt(prompt_b))
        except FileNotFoundError:
            pass
        sys.modules['streamlit'] = st
        sys.modules['flux_local'] = fl
        import app as app2
        app2._curation_add(1, app2.st.session_state.cur_batch[0])
        rows_b = dataset_rows_for_prompt(app2.base_prompt)
        self.assertGreaterEqual(rows_b, 1)
        # isolation check
        self.assertNotEqual(dataset_path_for_prompt(prompt_a), dataset_path_for_prompt(prompt_b))


if __name__ == '__main__':
    unittest.main()

