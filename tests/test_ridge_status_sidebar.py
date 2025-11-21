import sys
import types
import unittest


class _DummyFuture:
    def __init__(self, done_val: bool):
        self._done = done_val

    def done(self):
        return self._done


class TestRidgeStatusSidebar(unittest.TestCase):
    def _stub_streamlit(self):
        from tests.helpers.st_streamlit import stub_with_writes

        st, writes = stub_with_writes()
        # Keep Debug off for this test
        st.sidebar.checkbox = staticmethod(lambda *a, **k: k.get("value", False))
        return st, writes

    def _install_stubs(self, st):
        # flux_local, persistence_ui minimal stubs
        sys.modules["flux_local"] = types.SimpleNamespace(set_model=lambda *a, **k: None)
        sys.modules["persistence_ui"] = types.SimpleNamespace(render_metadata_panel=lambda *a, **k: None)
        sys.modules["latent_opt"] = types.SimpleNamespace(state_summary=lambda _l: {"pairs_logged": 0, "choices_logged": 0})
        sys.modules["ui"] = types.SimpleNamespace(sidebar_metric_rows=lambda *a, **k: None)
        sys.modules["ui_sidebar_train"] = types.SimpleNamespace(
            render_train_results_panel=lambda *a, **k: ("n/a", "n/a", "n/a", "Ridge (ok, rows=0)", "ok")
        )
        sys.modules["persistence"] = types.SimpleNamespace(
            dataset_rows_for_prompt=lambda _p: 0,
            dataset_stats_for_prompt=lambda _p: {"pos": 0, "neg": 0, "d": 0, "recent_labels": []},
            get_dataset_for_prompt_or_session=lambda _p, _s: (None, None),
        )

    def test_status_ok_without_future(self):
        st, writes = self._stub_streamlit()
        self._install_stubs(st)
        from constants import Keys

        class L:
            d = 4

        import ui_sidebar

        # Without a future, sidebar should show ok
        ui_sidebar.render_sidebar_tail(
            st,
            L(),
            "p",
            "state.npz",
            "Ridge",
            3,
            0.1,
            "dummy/model",
            lambda *a, **k: None,
            lambda *a, **k: None,
        )
        self.assertTrue(any("Ridge training: ok" in w for w in writes))


if __name__ == "__main__":
    unittest.main()
