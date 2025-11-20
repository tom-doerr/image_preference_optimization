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


class TestAsyncQueueMultiple(unittest.TestCase):
    def test_queue_fills_multiple_items(self):
        st = stub_basic()

        # Ensure Async queue mode selected and queue size small
        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, options, index=0):
                if "Generation mode" in label:
                    return "Async queue"
                if "Value model" in label:
                    return "Ridge"
                return options[index]

            @staticmethod
            def slider(label, *a, **k):
                if "Queue size" in label:
                    return 3
                return k.get("value", 1)

        st.sidebar = SB()
        sys.modules["streamlit"] = st

        # Stub background to return immediate futures
        bg = types.ModuleType("background")

        def schedule_decode_latents(*a, **k):
            return _ImmediateFuture("ok-image")

        bg.schedule_decode_latents = schedule_decode_latents
        bg.reset_executor = lambda *a, **k: None
        sys.modules["background"] = bg

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        if "app" in sys.modules:
            del sys.modules["app"]
        import app

        # After import, async queue mode runs; queue should be filled to size
        q = app.st.session_state.get("queue") or []
        self.assertGreaterEqual(len(q), 3)


if __name__ == "__main__":
    unittest.main()
