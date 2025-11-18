import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_basic
from persistence import dataset_rows_for_prompt


class _ImmediateFuture:
    def __init__(self, value):
        self._v = value
    def done(self):
        return True
    def result(self):
        return self._v


class TestAsyncQueueRejectAdvances(unittest.TestCase):
    def test_reject_advances_and_appends_dataset(self):
        st = stub_basic()
        st.session_state.prompt = 'asyncq_reject_unit_test'

        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, options, index=0):
                if 'Generation mode' in label:
                    return 'Async queue'
                if 'Value model' in label:
                    return 'Ridge'
                return options[index]
            @staticmethod
            def slider(label, *a, **k):
                if 'Queue size' in label:
                    return 3
                return k.get('value', 1)
        st.sidebar = SB()
        sys.modules['streamlit'] = st

        # Background: immediate futures
        bg = types.ModuleType('background')
        bg.reset_executor = lambda *a, **k: None
        bg.schedule_decode_latents = lambda *a, **k: _ImmediateFuture('ok-image')
        sys.modules['background'] = bg

        # Flux stub
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        # Import → prefill
        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        q = app.st.session_state.get('queue') or []
        self.assertGreaterEqual(len(q), 1)
        z0 = np.copy(q[0]['z'])
        rows_before = dataset_rows_for_prompt(st.session_state.prompt)

        # Click Reject 0
        st.button = lambda label, *a, **k: label == 'Reject 0'
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app2  # noqa: F401

        q2 = app.st.session_state.get('queue') or []
        self.assertGreaterEqual(len(q2), 1)
        self.assertGreater(float(np.linalg.norm(z0 - q2[0]['z'])), 1e-9)
        rows_after = dataset_rows_for_prompt(st.session_state.prompt)
        self.assertEqual(rows_after, rows_before + 1)


if __name__ == '__main__':
    unittest.main()

