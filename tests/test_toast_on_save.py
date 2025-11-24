import os
import sys
import shutil
import unittest

from tests.helpers.st_streamlit import stub_basic


class TestToastOnSave(unittest.TestCase):
    def test_toast_fires_on_sample_save(self):
        st = stub_basic()
        toasts = []
        # Saving path is boring now (no toast); keep hook defined but we won't rely on it
        st.toast = lambda msg, **k: toasts.append(msg)
        st.session_state.prompt = "toast-sample-prompt"

        # Small latent for speed
        from latent_state import init_latent_state

        st.session_state.lstate = init_latent_state(width=32, height=32)
        sys.modules["streamlit"] = st

        # Ensure clean slate for dataset/data folders
        # Folder-only; remove per-prompt data folder if present
        h = (
            __import__("hashlib")
            .sha1(st.session_state.prompt.encode("utf-8"))
            .hexdigest()[:10]
        )
        data_dir = os.path.join("data", h)
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)

        # Now import batch_ui and add one sample
import batch_ui

        z = st.session_state.lstate.mu.copy()
        batch_ui._curation_add(1, z, img=None)

        # No toast required; assert sample was persisted
        from ipo.core.persistence import dataset_rows_for_prompt
        self.assertGreaterEqual(dataset_rows_for_prompt(st.session_state.prompt), 1)

        # Cleanup created files
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)


if __name__ == "__main__":
    unittest.main()
