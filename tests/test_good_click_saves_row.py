import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestGoodClickSavesRow(unittest.TestCase):
    def test_good_click_appends_dataset_row(self):
        st = stub_basic()
        prompt = "good-click-saves"
        st.text_input = lambda *_, value="": prompt
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        # First render: capture key for Good (+1) 0
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

        # Second render: click Good (+1) 0 by returning True for its key
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

