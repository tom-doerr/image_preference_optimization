import sys
import types
import unittest

from tests.helpers.st_streamlit import stub_with_writes


class SidebarDimMismatchWarningTest(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar_extra", "streamlit"):
            sys.modules.pop(m, None)

    def test_warning_shows_when_dim_mismatch(self):
        st, writes = stub_with_writes()
        st.session_state.prompt = "p"
        st.session_state.dataset_dim_mismatch = (16384, 25600)
        st.session_state.lstate = types.SimpleNamespace(d=25600)
        sys.modules["streamlit"] = st

        import ui_sidebar_extra

        ui_sidebar_extra.render_rows_and_last_action(st, st.session_state.prompt, st.session_state.lstate)
        out = "\n".join(writes)
        self.assertIn("Dataset recorded at d=16384 (ignored); current latent dim d=25600", out)


if __name__ == "__main__":
    unittest.main()
