import sys
import types
import unittest

import numpy as np

from constants import Keys
from tests.helpers.st_streamlit import stub_basic


class _FragCounter:
    def __init__(self):
        self.calls = 0

    def __call__(self, fn):
        def _w(*a, **k):
            self.calls += 1
            return fn(*a, **k)

        return _w


class TestFragmentsDefaultAndToggle(unittest.TestCase):
    def setUp(self):
        # Common fast stubs
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

    def tearDown(self):
        for m in ("batch_ui", "streamlit", "flux_local"):
            sys.modules.pop(m, None)

    def _minimal_state(self, st):
        # Minimal lstate and batch to avoid latent_opt imports
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4, sigma=1.0, w=np.zeros(4), rng=np.random.default_rng(0)
        )
        st.session_state.cur_batch = [np.zeros(4) for _ in range(2)]
        st.session_state.cur_labels = [None, None]

    def test_fragments_default_on(self):
        st = stub_basic()
        self._minimal_state(st)
        frag = _FragCounter()
        st.fragment = frag
        sys.modules["streamlit"] = st
        import batch_ui

        batch_ui._render_batch_ui()
        self.assertGreater(frag.calls, 0)

    def test_fragments_can_be_disabled_via_key(self):
        st = stub_basic()
        self._minimal_state(st)
        frag = _FragCounter()
        st.fragment = frag
        st.session_state[Keys.USE_FRAGMENTS] = False
        sys.modules["streamlit"] = st
        import batch_ui

        batch_ui._render_batch_ui()
        self.assertEqual(frag.calls, 0)

    def test_tile_cache_populated_with_fragments(self):
        st = stub_basic()
        self._minimal_state(st)
        frag = _FragCounter()
        st.fragment = frag
        sys.modules["streamlit"] = st
        import batch_ui

        batch_ui._render_batch_ui()
        cache = st.session_state.get("_tile_cache", {})
        self.assertTrue(isinstance(cache, dict))
        self.assertGreater(len(cache), 0)


if __name__ == "__main__":
    unittest.main()

