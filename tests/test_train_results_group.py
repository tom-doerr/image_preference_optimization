import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestTrainResultsGroup(unittest.TestCase):
    def test_group_expander_called_and_lines_present(self):
        st, writes = stub_with_writes()
        st.session_state.prompt = "train-results-group"

        labels = []

        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def expander(label, expanded=False):
            labels.append(label)
            return Ctx()

        st.sidebar.expander = expander
        sys.modules["streamlit"] = st

        fl = types.ModuleType("flux_local")
        fl.set_model = lambda *a, **k: None
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        # Import the app to trigger sidebar tail/expander emission under this stub
        import app  # noqa: F401

        out = "\n".join(writes)
        self.assertIn("Train results", labels)
        self.assertIn("Train score:", out)
        self.assertIn("CV score:", out)
        self.assertIn("Last CV:", out)
        self.assertIn("Last train:", out)


if __name__ == "__main__":
    unittest.main()
