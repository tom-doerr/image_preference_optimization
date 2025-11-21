import sys
import types
import unittest
import numpy as np

from tests.helpers.st_streamlit import stub_basic


class DummyFuture:
    def __init__(self):
        self._cancelled = False
    def cancel(self):
        self._cancelled = True
        return True
    def done(self):
        return False


class TestXgbCancelButton(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar", "streamlit", "value_model", "persistence", "flux_local", "persistence_ui"):
            sys.modules.pop(m, None)

    def test_cancel_button_clears_future_and_sets_status(self):
        st = stub_basic()
        # Make the cancel button return True
        def _button(label, *a, **k):
            return label == "Cancel current XGB fit"

        st.sidebar.button = staticmethod(_button)
        st.session_state.prompt = "p"
        lstate = types.SimpleNamespace(d=4)
        st.session_state.lstate = lstate
        st.session_state.xgb_fit_future = DummyFuture()

        # Provide a small dataset so rows_now > 0
        P = types.ModuleType("persistence")
        P.get_dataset_for_prompt_or_session = lambda *a, **k: (
            np.ones((3, 4), dtype=float),
            np.array([+1.0, -1.0, +1.0], dtype=float),
        )
        P.dataset_rows_for_prompt = lambda *a, **k: 3
        P.dataset_stats_for_prompt = lambda *a, **k: {"rows": 3, "pos": 2, "neg": 1, "d": 4, "recent_labels": [1, -1]}
        sys.modules["persistence"] = P

        VM = types.ModuleType("value_model")
        VM.ensure_fitted = lambda *a, **k: None
        sys.modules["value_model"] = VM

        FL = types.ModuleType("flux_local")
        FL.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = FL
        PUI = types.ModuleType("persistence_ui")
        PUI.render_metadata_panel = lambda *a, **k: None
        sys.modules["persistence_ui"] = PUI

        sys.modules["streamlit"] = st

        import ui_sidebar

        ui_sidebar.render_sidebar_tail(
            st,
            lstate,
            st.session_state.prompt,
            state_path="/tmp/x",
            vm_choice="XGBoost",
            iter_steps=1,
            iter_eta=None,
            selected_model="stabilityai/sd-turbo",
            apply_state_cb=lambda *a, **k: None,
            rerun_cb=lambda *a, **k: None,
        )

        # Future cleared and status set to cancelled
        self.assertIsNone(st.session_state.get("xgb_fit_future"))
        self.assertIn("xgb_train_status", st.session_state)
        self.assertEqual(st.session_state["xgb_train_status"].get("state"), "cancelled")


if __name__ == "__main__":
    unittest.main()

