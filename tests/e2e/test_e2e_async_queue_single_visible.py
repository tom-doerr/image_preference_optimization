import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_capture_images


class _ImmediateFuture:
    def __init__(self, value):
        self._v = value
    def done(self):
        return True
    def result(self):
        return self._v


class TestE2EAsyncQueueSingleVisible(unittest.TestCase):
    def test_only_head_item_is_rendered(self):
        st, images = stub_capture_images()
        # Async queue mode, size 3
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

        # Background stub: immediate completion
        bg = types.ModuleType('background')
        bg.reset_executor = lambda *a, **k: None
        bg.schedule_decode_latents = lambda *a, **k: _ImmediateFuture('ok-image')
        sys.modules['background'] = bg

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401

        # Only one queue image (Item 0) should render
        item_caps = [c for c in images if c.startswith('Item ')]
        self.assertEqual(len(item_caps), 1)
        self.assertIn('Item 0', item_caps[0])


if __name__ == '__main__':
    unittest.main()

