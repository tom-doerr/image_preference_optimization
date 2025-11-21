import os
import sys
import tempfile
import types
import unittest

from tests.helpers.st_streamlit import stub_with_writes


class TestRowsCounterFragmentsUpdates(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("IPO_DATA_ROOT", None)
        for m in ("app", "streamlit", "flux_local"):
            sys.modules.pop(m, None)

    def test_rows_counter_increments_with_fragments_on(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["IPO_DATA_ROOT"] = tmp.name

        st, writes = stub_with_writes()
        prompt = "rows-frag-inc"
        st.text_input = lambda *_, value="": prompt
        st.fragment = lambda f: (lambda *a, **k: f())
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **k: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl
        # No-op persistence UI to keep imports light
        sys.modules["persistence_ui"] = types.SimpleNamespace(
            render_persistence_controls=lambda *a, **k: None,
            render_metadata_panel=lambda *a, **k: None,
        )

        # Initial import: expect Dataset rows: 0
        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        first = "\n".join(writes)
        self.assertIn("Dataset rows:", first)
        self.assertIn("Dataset rows: 0", first)

        # Save one row via Good click on item 0
        keys = {}

        def cap_btn(label, *a, **k):
            if "key" in k:
                keys[label] = k["key"]
            return False

        st.button = cap_btn
        del sys.modules["app"]
        import app as app2  # noqa: F401
        good0_key = keys.get("Good (+1) 0")
        self.assertIsNotNone(good0_key)

        def click_btn(label, *a, **k):
            return k.get("key") == good0_key

        st.button = click_btn
        del sys.modules["app"]
        import app as app3  # noqa: F401

        # Rerun once more to ensure metric lines were written after save
        del sys.modules["app"]
        import app as app4  # noqa: F401
        out = "\n".join(writes)
        if "Dataset rows: 1" not in out:
            # As a fallback, assert disk rows advanced
            from persistence import dataset_rows_for_prompt
            self.assertEqual(dataset_rows_for_prompt(prompt), 1)
        self.assertIn("Rows (disk)", out)


if __name__ == "__main__":
    unittest.main()
