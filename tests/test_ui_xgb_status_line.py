import sys
import types
import unittest

import numpy as np
from constants import Keys
from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarXGBStatus(unittest.TestCase):
    def tearDown(self):
        for m in ("streamlit", "ui_sidebar", "value_model"):
            sys.modules.pop(m, None)

    def test_xgb_status_line_flips_after_sync_fit(self):
        st, writes = stub_with_writes()

        # Provide a minimal expander so Details block renders
        class _CM:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        st.sidebar.expander = lambda *a, **k: _CM()

        # Minimal session state
        st.session_state[Keys.VM_CHOICE] = "XGBoost"
        st.session_state.prompt = "xgb-status-test"

        # Latent state with a tiny in-memory dataset (used by sync fit button)
        lstate = types.SimpleNamespace(d=4, w=np.zeros(4), X=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], float), y=np.array([1.0, -1.0, 1.0]))

        # Seed a running status
        st.session_state[Keys.XGB_TRAIN_STATUS] = {"state": "running", "rows": 0, "lam": 1.0}

        # Stub value_model.fit_value_model to perform a synchronous "fit"
        vm = types.ModuleType("value_model")

        def fit_value_model(vm_choice, _lstate, X, y, lam, session_state):
            session_state[Keys.XGB_CACHE] = {"model": "stub", "n": int(getattr(X, "shape", (0,))[0])}
            session_state[Keys.XGB_TRAIN_STATUS] = {"state": "ok", "rows": int(X.shape[0]), "lam": float(lam)}

        def ensure_fitted(*a, **k):
            return None

        vm.fit_value_model = fit_value_model
        vm.ensure_fitted = ensure_fitted
        sys.modules["value_model"] = vm

        # First render: no click; should show running status with 0 rows
        sys.modules["streamlit"] = st
        # Stub flux_local.set_model so no heavy imports happen
        fl = types.ModuleType("flux_local")
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl
        import ipo.ui.ui_sidebar as ui_sidebar

        ui_sidebar.render_sidebar_tail(
            st,
            lstate,
            st.session_state.prompt,
            "data/dda60c595a/latent_state.npz",
            "XGBoost",
            10,
            0.1,
            "stabilityai/sd-turbo",
            lambda *a, **k: None,
            lambda *a, **k: None,
        )
        self.assertTrue(any("XGBoost model rows: 0 (status: running)" in w for w in writes))

        # Second render: click the sync-fit button once
        def _button(label, *a, **k):
            return label == "Train XGBoost now (sync)"

        st.sidebar.button = _button
        writes.clear()
        ui_sidebar.render_sidebar_tail(
            st,
            lstate,
            st.session_state.prompt,
            "data/dda60c595a/latent_state.npz",
            "XGBoost",
            10,
            0.1,
            "stabilityai/sd-turbo",
            lambda *a, **k: None,
            lambda *a, **k: None,
        )

        self.assertTrue(any("XGBoost model rows: 3 (status: ok)" in w for w in writes))


if __name__ == "__main__":
    unittest.main()
