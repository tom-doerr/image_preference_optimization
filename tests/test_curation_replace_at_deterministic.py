import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_basic


class TestCurationReplaceAtDeterministic(unittest.TestCase):
    def test_replace_at_is_deterministic_and_preserves_size(self):
        st = stub_basic()
        st.text_input = lambda *_, value="": "replace-at-deterministic"
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        if "app" in sys.modules:
            del sys.modules["app"]
        import app

        app._curation_init_batch()
        batch0 = [np.copy(z) for z in st.session_state.cur_batch]
        n0 = len(batch0)

        app._curation_replace_at(1)
        batch1 = st.session_state.cur_batch
        self.assertEqual(len(batch1), n0)
        # Only index 1 should differ
        diffs = [not np.allclose(batch0[i], batch1[i]) for i in range(n0)]
        self.assertEqual(sum(diffs), 1)
        self.assertTrue(diffs[1])

        # Calling replace_at(1) again yields the same vector (deterministic)
        prev = np.copy(batch1[1])
        app._curation_replace_at(1)
        self.assertTrue(np.allclose(prev, st.session_state.cur_batch[1]))


if __name__ == "__main__":
    unittest.main()

