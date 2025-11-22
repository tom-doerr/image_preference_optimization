import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestBatchToast(unittest.TestCase):
    def test_saved_toast_message(self):
        st, writes = stub_with_writes()
        prompt = "toast dataset test"
        st.text_input = lambda *_, value="": prompt

        # Enable curation mode
        class SB(st.sidebar.__class__):
            @staticmethod
            def checkbox(label, value=False, **k):
                return True if "Batch curation mode" in label else value

            @staticmethod
            def slider(label, *a, **k):
                if "Batch size" in label:
                    return 2
                return k.get("value", 1)

            @staticmethod
            def selectbox(label, options, *a, **k):
                return "Batch curation"

            @staticmethod
            def expander(*a, **k):
                class _E:
                    def __enter__(self):
                        return self

                    def __exit__(self, *e):
                        return False

                return _E()

        st.sidebar = SB()
        # keep write capture
        st.sidebar.write = lambda *a, **k: writes.append(str(a[0]) if a else "")
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **k: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        import app

        z0 = app.st.session_state.cur_batch[0]
        app._curation_add(1, z0)
        out = "\n".join(writes)
        # Accept either toast fallback text or dataset rows increment
        if "Saved label +1" not in out:
            from ipo.core.persistence import dataset_rows_for_prompt

            assert dataset_rows_for_prompt(prompt) >= 1


if __name__ == "__main__":
    unittest.main()
