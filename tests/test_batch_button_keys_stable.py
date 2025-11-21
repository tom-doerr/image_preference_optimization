import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestBatchButtonKeysStable(unittest.TestCase):
    def test_good_bad_keys_stable_across_reruns(self):
        st = stub_basic()
        prompt = "button-keys-stable"
        st.text_input = lambda *_, value="": prompt
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        # Capture keys during first render
        keys1 = []
        def cap_btn(label, *a, **k):
            if "key" in k:
                keys1.append((label, k["key"]))
            return False

        st.button = cap_btn
        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        good0_key_1 = next((k for (lbl, k) in keys1 if lbl.startswith("Good (+1) 0")), None)
        bad0_key_1 = next((k for (lbl, k) in keys1 if lbl.startswith("Bad (-1) 0")), None)
        self.assertIsNotNone(good0_key_1)
        self.assertIsNotNone(bad0_key_1)

        # Rerun: keys should match
        keys2 = []
        def cap_btn2(label, *a, **k):
            if "key" in k:
                keys2.append((label, k["key"]))
            return False

        st.button = cap_btn2
        if "app" in sys.modules:
            del sys.modules["app"]
        import app as app2  # noqa: F401

        good0_key_2 = next((k for (lbl, k) in keys2 if lbl.startswith("Good (+1) 0")), None)
        bad0_key_2 = next((k for (lbl, k) in keys2 if lbl.startswith("Bad (-1) 0")), None)
        self.assertEqual(good0_key_1, good0_key_2)
        self.assertEqual(bad0_key_1, bad0_key_2)


if __name__ == "__main__":
    unittest.main()

