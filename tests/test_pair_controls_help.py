import sys
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestPairControlsHelp(unittest.TestCase):
    def test_explanatory_line_present(self):
        st, writes = stub_with_writes()
        st.session_state.prompt = "pair-help"

        # Force the expander path to run while keeping write capture
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        st.sidebar.expander = lambda *a, **k: Ctx()
        sys.modules["streamlit"] = st
        # Import just the control builder
        from ui_controls import build_pair_controls

        build_pair_controls(st, expanded=True)
        text = "\n".join(writes)
        self.assertIn("Alpha scales d1", text)
        self.assertIn("Trust radius", text)


if __name__ == "__main__":
    unittest.main()
