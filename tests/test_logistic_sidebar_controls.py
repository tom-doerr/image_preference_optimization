import sys
import types
import unittest

from tests.helpers.st_streamlit import stub_with_writes


class _LState:
    def __init__(self, d=2):
        import numpy as np

        self.d = d
        self.mu = np.zeros(d, dtype=float)
        self.w = np.zeros(d, dtype=float)
        self.width = 640
        self.height = 640


class TestLogisticSidebarControls(unittest.TestCase):
    def test_controls_create_session_keys(self):
        st, writes = stub_with_writes()
        # Force Value model selectbox to return 'Logistic'
        st.sidebar.selectbox = staticmethod(lambda label, options, index=0: "Logistic" if label == "Value model" else options[index])
        # Stub flux_local
        fl = types.ModuleType("flux_local")
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl
        # Minimal latent_opt import surface used by sidebar tail
        import latent_opt as _lat
        # Build UI
        from ui_sidebar import render_modes_and_value_model, render_sidebar_tail

        vm_choice, _mode, _bs, _ = render_modes_and_value_model(st)
        self.assertEqual(vm_choice, "Logistic")
        lstate = _LState(d=2)
        render_sidebar_tail(
            st,
            lstate,
            prompt="p",
            state_path="latent_state_dummy.npz",
            vm_choice=vm_choice,
            iter_steps=5,
            iter_eta=0.001,
            selected_model="stabilityai/sd-turbo",
            apply_state_cb=lambda *a, **k: None,
            rerun_cb=lambda *a, **k: None,
        )
        from constants import Keys

        # Keys must be present with defaults
        self.assertIn(Keys.LOGIT_STEPS, st.session_state)
        self.assertIn(Keys.LOGIT_L2, st.session_state)


if __name__ == "__main__":
    unittest.main()

