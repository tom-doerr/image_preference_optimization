import sys
import types
import unittest
import numpy as np

from tests.helpers.st_streamlit import stub_basic


class TestXgbTrainNowButton(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar", "streamlit", "value_model", "persistence", "flux_local", "persistence_ui"):
            sys.modules.pop(m, None)

    def test_click_runs_sync_fit(self):
        st = stub_basic()
        # Make the button return True for our label
        def _button(label, *a, **k):
            return label == "Train XGBoost now (sync)"

        st.sidebar.button = staticmethod(_button)
        st.session_state.prompt = "p"
        lstate = types.SimpleNamespace(d=4)
        st.session_state.lstate = lstate

        # Provide a small balanced dataset
        P = types.ModuleType("persistence")
        P.get_dataset_for_prompt_or_session = lambda *a, **k: (
            np.ones((4, 4), dtype=float),
            np.array([+1.0, -1.0, +1.0, -1.0], dtype=float),
        )
        P.dataset_rows_for_prompt = lambda *a, **k: 4
        P.dataset_stats_for_prompt = lambda *a, **k: {"rows": 4, "pos": 2, "neg": 2, "d": 4, "recent_labels": [1, -1]}
        sys.modules["persistence"] = P

        # fit_value_model stub that records calls and marks a cache
        called = {"n": 0}

        def _fit_vm(vm_choice, lstate, X, y, lam, ss):
            called["n"] += 1
            ss.xgb_cache = {"model": object(), "n": int(getattr(X, "shape", (0,))[0])}
            ss.xgb_train_status = {"state": "ok", "rows": int(X.shape[0]), "lam": float(lam)}

        VM = types.ModuleType("value_model")
        VM.fit_value_model = _fit_vm
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

        self.assertEqual(called["n"], 1)


if __name__ == "__main__":
    unittest.main()

