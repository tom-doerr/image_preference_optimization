import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestAsyncQueueBasic(unittest.TestCase):
    def test_queue_fills_and_labels(self):
        st = stub_basic(pre_images=False)

        # Enable async queue and set sizes
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if "Async queue mode" in label else value

            @staticmethod
            def slider(label, *a, **k):
                if "Queue size" in label:
                    return 3
                return k.get("value", 1)

            @staticmethod
            def selectbox(label, options, *a, **k):
                return "Async queue"

            @staticmethod
            def expander(*a, **k):
                class _E:
                    def __enter__(self):
                        return self

                    def __exit__(self, *e):
                        return False

                return _E()

        st.sidebar = SB()
        sys.modules["streamlit"] = st

        # Fast generator stubs
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **k: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        import app

        # With single-image background, queue fills gradually; call a few times
        for _ in range(3):
            app._queue_fill_up_to()
        q = list(app.st.session_state.get("queue") or [])
        n_before = len(q)
        self.assertGreaterEqual(n_before, 1)
        # Label first item and ensure it is removed
        app._queue_label(0, 1)
        q2 = list(app.st.session_state.get("queue") or [])
        self.assertLess(len(q2), n_before)


if __name__ == "__main__":
    unittest.main()
