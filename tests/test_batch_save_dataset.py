import os
import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic
from persistence import dataset_path_for_prompt


class TestBatchSaveDataset(unittest.TestCase):
    def test_accept_reject_persist_to_file(self):
        st = stub_basic(pre_images=False)
        # Use a unique prompt so we don't collide with other tests/files
        prompt = 'batch save dataset test'
        st.text_input = lambda *_, value="": prompt

        # Enable batch curation mode with small batch
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'Batch curation mode' in label else value
            @staticmethod
            def slider(label, *a, **k):
                if 'Batch size' in label:
                    return 2
                if 'Ridge' in label:
                    return 0.01
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

        # Ensure a clean dataset file
        path = dataset_path_for_prompt(prompt)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

        import app

        # Accept first, Reject second -> dataset should gain 2 rows and labels [1,-1]
        z0, z1 = app.st.session_state.cur_batch[:2]
        app._curation_add(1, z0)
        app._curation_add(-1, z1)
        from persistence import dataset_rows_for_prompt
        self.assertGreaterEqual(dataset_rows_for_prompt(prompt), 2)

        # Add one more and ensure it appends to 3 rows
        app._curation_add(1, app.st.session_state.cur_batch[0])
        self.assertGreaterEqual(dataset_rows_for_prompt(prompt), 3)


if __name__ == '__main__':
    unittest.main()
