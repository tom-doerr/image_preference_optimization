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

        # Initialize minimal latent state + prompt without importing the full app
        from latent_state import init_latent_state
        st.session_state.lstate = init_latent_state()
        st.session_state.prompt = 'toast-queue'
        # Seed a one-item queue and label it (z can be zeros of the right dim)
        import numpy as np
        z0 = np.zeros(st.session_state.lstate.d, dtype=float)
        st.session_state.queue = [{'z': z0, 'future': DummyFuture(), 'label': None}]
        import queue_ui
        queue_ui._queue_label(0, 1, img='ok-image')
        out = "\n".join(writes)
        assert ('Accepted (+1)' in out) or ('Saved sample #' in out), out


if __name__ == '__main__':
    unittest.main()
