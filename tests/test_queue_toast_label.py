import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class DummyFuture:
    def result(self):
        return 'ok-image'


class TestQueueToast(unittest.TestCase):
    def test_accept_reject_toasts(self):
        st, writes = stub_with_writes()
        # Make toast write into our capture so we can assert on it
        st.toast = lambda msg: st.sidebar.write(str(msg))
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        # Import app to init session state/lstate
        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401

        # Seed a one-item queue and label it
        st.session_state.queue = [{'z': app.st.session_state.cur_batch[0], 'future': DummyFuture(), 'label': None}]
        import queue_ui
        queue_ui._queue_label(0, 1, img='ok-image')
        out = "\n".join(writes)
        self.assertIn('Accepted (+1)', out)


if __name__ == '__main__':
    unittest.main()

