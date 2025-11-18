import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_main_writes


class _PendingFuture:
    def done(self):
        return False
    def result(self):
        raise RuntimeError('should not be called when not done')


class TestAsyncQueueLoadingState(unittest.TestCase):
    def test_loading_message_when_future_not_done(self):
        st, writes = stub_with_main_writes()
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
                    return 2
                return k.get('value', 1)
        st.sidebar = SB()
        sys.modules['streamlit'] = st

        # Background: pending futures (not done)
        bg = types.ModuleType('background')
        bg.reset_executor = lambda *a, **k: None
        bg.schedule_decode_latents = lambda *a, **k: _PendingFuture()
        sys.modules['background'] = bg

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401

        text = "\n".join(writes)
        self.assertIn('Item 0: loading…', text)


if __name__ == '__main__':
    unittest.main()

