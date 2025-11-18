import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class _ImmediateFuture:
    def __init__(self, value):
        self._v = value
    def done(self):
        return True
    def result(self):
        return self._v


class TestAsyncQueueRefillSize(unittest.TestCase):
    def test_refills_to_queue_size_after_accept(self):
        st = stub_basic()
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

        bg = types.ModuleType('background')
        bg.reset_executor = lambda *a, **k: None
        bg.schedule_decode_latents = lambda *a, **k: _ImmediateFuture('ok-image')
        sys.modules['background'] = bg

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        # First import → filled
        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        self.assertGreaterEqual(len(app.st.session_state.get('queue') or []), 3)
        # Accept head
        st.button = lambda label, *a, **k: label == 'Accept 0'
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app2  # noqa: F401
        # Reimport one more time so fill_up_to runs again
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app3
        self.assertGreaterEqual(len(app3.st.session_state.get('queue') or []), 3)


if __name__ == '__main__':
    unittest.main()

