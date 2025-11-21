import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_with_writes


class TestUiSidebarXgbSyncStubbed(unittest.TestCase):
    def setUp(self):
        for m in (
            "ui_sidebar",
            "streamlit",
            "value_model",
            "persistence",
            "latent_opt",
            "flux_local",
            "value_scorer",
        ):
            sys.modules.pop(m, None)

    def _expander_ctx(self):
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Ctx()

    def test_train_now_sync_sets_cache_and_status_ok(self):
        st, writes = stub_with_writes()
        st.session_state.prompt = "xgb-sync-test"
        st.sidebar.expander = lambda *a, **k: self._expander_ctx()
        # No train click; simulate a ready cached model instead
        st.sidebar.button = lambda *a, **k: False
        sys.modules["streamlit"] = st

        # Provide a tiny in-memory dataset with both classes
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        y = np.array([+1.0, -1.0], dtype=float)
        lstate = types.SimpleNamespace(X=X, y=y, d=2, w=np.zeros(2))

        # Stub modules used by the sidebar
        sys.modules["flux_local"] = types.SimpleNamespace(
            set_model=lambda *a, **k: None,
            get_last_call=lambda: {},
        )
        # Persistence not needed since we pass in-memory X/y, but keep surface
        sys.modules["persistence"] = types.SimpleNamespace(
            dataset_rows_for_prompt=lambda p: 0,
            dataset_stats_for_prompt=lambda p: {"rows": 0, "pos": 0, "neg": 0, "d": 0},
            get_dataset_for_prompt_or_session=lambda p, ss: (None, None),
            read_metadata=lambda *a, **k: {},
        )
        # Provide a pre-ready cache to exercise UI paths
        st.session_state.xgb_cache = {"model": object(), "n": int(X.shape[0])}
        sys.modules["value_model"] = types.SimpleNamespace(fit_value_model=lambda *a, **k: None)
        # Scorer pulls from session_state.xgb_cache → ok after fit
        import value_scorer as _vs  # real module is fine
        sys.modules["value_scorer"] = _vs
        # Minimal latent_opt surface
        sys.modules["latent_opt"] = types.SimpleNamespace(
            state_summary=lambda *_: {"pairs_logged": 0, "choices_logged": 0}
        )

        import ui_sidebar as u

        u.render_sidebar_tail(
            st,
            lstate,
            st.session_state.prompt,
            "latent_state_test.npz",
            "XGBoost",
            0,
            0.0,
            "stabilityai/sd-turbo",
            lambda *a, **k: None,
            lambda *a, **k: None,
        )
        text = "\n".join(writes)
        # Status flips to ok with a cached model
        self.assertIn("Value scorer status: ok", text)
        # No extra status lines printed (216g)
        self.assertNotIn("XGBoost training:", text)
        self.assertNotIn("Ridge training:", text)


if __name__ == "__main__":
    unittest.main()
