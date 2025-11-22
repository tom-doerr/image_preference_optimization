import sys
import types
import unittest

import numpy as np

from tests.helpers.st_streamlit import stub_basic


class TestRandomAnchorCheckbox(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar", "streamlit", "latent_logic"):
            sys.modules.pop(m, None)

    def test_checkbox_sets_flag_and_state_and_affects_anchor(self):
        st = stub_basic()
        # Force the checkbox to return True for our label
        def _checkbox(label, *a, **k):
            return label == "Use random anchor (ignore prompt)"

        st.sidebar.checkbox = _checkbox  # type: ignore[attr-defined]
        # Minimal latent state in session
        st.session_state.lstate = types.SimpleNamespace(d=8, sigma=1.0, rng=np.random.default_rng(0))
        # Install stubs
        sys.modules["streamlit"] = st
        # Import UI and latent helpers
        import ipo.ui.ui_sidebar as ui_sidebar
        import latent_logic as ll

        # Act: render the Mode & value model block (emits the checkbox)
        ui_sidebar.render_modes_and_value_model(st)

        # Assert: session flag and state attribute are set
        self.assertTrue(st.session_state.get("use_random_anchor"))
        self.assertTrue(getattr(st.session_state.lstate, "use_random_anchor", False))

        # With random anchor on, z_from_prompt should ignore prompt text
        z1 = ll.z_from_prompt(st.session_state.lstate, "p1")
        z2 = ll.z_from_prompt(st.session_state.lstate, "p2")
        self.assertEqual(z1.shape[0], z2.shape[0])
        self.assertTrue(np.allclose(z1, z2), "random anchor should ignore prompt differences")


if __name__ == "__main__":
    unittest.main()
