import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarLastTrain(unittest.TestCase):
    def test_last_train_shows_after_training(self):
        st, writes = stub_with_writes()
        # Enable batch mode and trigger Good + Keep batch training
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
        # capture writes/metrics
        st.sidebar.write = lambda *a, **k: writes.append(str(a[0]) if a else "")
        st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
        def _btn(label, *a, **k):
            return label in ('Good (+1) 0', 'Train on dataset (keep batch)')
        st.button = _btn
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **kw: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        # First import performs labeling and training
        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401
        # Re-import to re-render Data section with updated timestamp
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app2  # noqa: F401

        out = "\n".join(writes)
        # Should include a Last train line; the latest occurrence should be a timestamp
        self.assertIn('Last train:', out)
        last_lines = [ln for ln in out.splitlines() if ln.startswith('Last train:')]
        self.assertTrue(last_lines, 'no Last train lines found')
        self.assertNotEqual(last_lines[-1].strip(), 'Last train: n/a')


if __name__ == '__main__':
    unittest.main()
