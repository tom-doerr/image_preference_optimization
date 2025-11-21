import sys
import types
import unittest
import numpy as np

from tests.helpers.st_streamlit import stub_basic


class TestFragmentsNoRedecodeOnRerender(unittest.TestCase):
    def tearDown(self):
        for m in ("batch_ui", "streamlit", "flux_local"):
            sys.modules.pop(m, None)

    def test_no_second_decode_when_cached(self):
        st = stub_basic()
        st.fragment = lambda f: (lambda *a, **k: f())
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4, sigma=1.0, w=np.ones(4), rng=np.random.default_rng(0)
        )
        st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
        st.session_state.cur_labels = [None, None]
        sys.modules["streamlit"] = st

        calls = {"n": 0}

        def _gen(*a, **k):
            calls["n"] += 1
            return "ok-image"

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = _gen
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        import batch_ui

        # First render: decodes both tiles
        batch_ui._render_batch_ui()
        n1 = calls["n"]
        self.assertGreaterEqual(n1, 2)

        # Second render: should reuse cached images and not decode again
        batch_ui._render_batch_ui()
        self.assertEqual(calls["n"], n1)


if __name__ == "__main__":
    unittest.main()

