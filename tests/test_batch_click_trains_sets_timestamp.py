import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestBatchClickTrainsTimestamp(unittest.TestCase):
    def test_good_click_sets_last_train(self):
        st, writes = stub_with_writes()
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if 'Batch curation mode' in label else value
            @staticmethod
            def slider(label, *a, **k):
                if 'Batch size' in label:
                    return 2
                return k.get('value', 1)
        st.sidebar = SB()
        st.sidebar.write = lambda *a, **k: writes.append(str(a[0]) if a else "")
        st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
        # Click Good on the first item
        st.button = lambda label, *a, **k: label == 'Good (+1) 0'
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        sys.modules['flux_local'] = fl

        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401
        # Re-import to re-render Data section with updated timestamp
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app2  # noqa: F401

        out = "\n".join(writes)
        self.assertIn('Last train:', out)
        last_lines = [ln for ln in out.splitlines() if ln.startswith('Last train:')]
        self.assertTrue(last_lines and last_lines[-1] != 'Last train: n/a')


if __name__ == '__main__':
    unittest.main()
