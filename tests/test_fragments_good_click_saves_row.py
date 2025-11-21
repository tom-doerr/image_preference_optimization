import os
import sys
import tempfile
import types
import unittest

import numpy as np

from tests.helpers.st_streamlit import stub_basic


class TestFragmentsGoodClickSavesRow(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("IPO_DATA_ROOT", None)
        for m in ("app", "streamlit", "flux_local"):
            sys.modules.pop(m, None)

    def test_good_click_saves_with_fragments_enabled(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["IPO_DATA_ROOT"] = tmp.name

        st = stub_basic()
        prompt = "fragments-good-click"
        st.text_input = lambda *_, value="": prompt
        # Enable fragments and provide a simple fragment() wrapper
        st.fragment = lambda f: (lambda *a, **k: f())
        sys.modules["streamlit"] = st

        # Stub flux_local minimal decode
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **k: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        # First render: capture the stable Good(+1) key for item 0
        keys = {}

        def cap_btn(label, *a, **k):
            if "key" in k:
                keys[label] = k["key"]
            return False

        st.button = cap_btn
        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        good0_key = keys.get("Good (+1) 0")
        self.assertIsNotNone(good0_key)

        # Second render: click the captured Good key
        def click_btn(label, *a, **k):
            return k.get("key") == good0_key

        st.button = click_btn
        del sys.modules["app"]
        sys.modules["streamlit"] = st
        sys.modules["flux_local"] = fl
        import app as app2  # noqa: F401

        # Assert dataset rows on disk is at least 1 for this prompt
        from persistence import dataset_rows_for_prompt

        self.assertGreaterEqual(dataset_rows_for_prompt(prompt), 1)


if __name__ == "__main__":
    unittest.main()

