import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic
from persistence import dataset_rows_for_prompt


class TestQueueModeSaveDataset(unittest.TestCase):
    def test_queue_accept_appends_row(self):
        st = stub_basic(pre_images=False)
        prompt = "queue mode save dataset test"
        st.text_input = lambda *_, value="": prompt

        class SB(st.sidebar.__class__):
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

            @staticmethod
            def slider(*a, **k):
                return k.get("value", 1)

            @staticmethod
            def checkbox(*a, **k):
                return False

        st.sidebar = SB()
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **k: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        # Clean file
        # no-op: folder data is unique per prompt

        import app
        from latent_state import init_latent_state

        app._apply_state(init_latent_state())
        app._queue_fill_up_to()
        # Accept first item
        app._queue_label(0, 1)
        self.assertGreaterEqual(dataset_rows_for_prompt(app.base_prompt), 1)


if __name__ == "__main__":
    unittest.main()
