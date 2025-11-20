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


class TestBatchUIBackgroundFallback(unittest.TestCase):
    def test_images_render_when_result_or_sync_missing(self):
        # Streamlit stub capturing image captions
        st, images = stub_capture_images()

        # Force Batch mode in sidebar
        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, options, *a, **k):
                if 'Generation mode' in label:
                    return 'Batch curation'
                return options[0] if options else ''

            @staticmethod
            def expander(*a, **k):
                class _E:
                    def __enter__(self): return self
                    def __exit__(self, *e): return False
                return _E()

        st.sidebar = SB()
        sys.modules['streamlit'] = st

        # Minimal flux_local stub (no real GPU usage)
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **k: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        # Background stub with simple helper used by batch_ui
        bg = types.ModuleType('background')
        bg.schedule_decode_latents = lambda *a, **k: _ImmediateFuture('ok-image')

        def _ros(fut, started_at, timeout_s, sync_callable):
            if fut is not None and fut.done():
                return fut.result(), fut
            return None, fut

        bg.result_or_sync_after = _ros
        sys.modules['background'] = bg

        # Import app to run the UI once
        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401

        # We should have rendered at least one batch item image with caption "Item 0"
        self.assertTrue(any(c.startswith('Item 0') for c in images))


if __name__ == '__main__':
    unittest.main()
