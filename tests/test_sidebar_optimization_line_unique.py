import sys
import types
import unittest

from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarOptimizationLineUnique(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar", "streamlit", "ui", "ui_metrics", "ui_controls", "flux_local", "persistence", "persistence_ui"):
            sys.modules.pop(m, None)

    def test_optimization_line_appears_twice_total(self):
        st, writes = stub_with_writes()
        sys.modules["streamlit"] = st
        sys.modules["ui"] = types.SimpleNamespace(sidebar_metric_rows=lambda *a, **k: None, sidebar_metric=lambda *a, **k: None, status_panel=lambda *a, **k: None)
        sys.modules["ui_metrics"] = types.SimpleNamespace(render_iter_step_scores=lambda *a, **k: None, render_mu_value_history=lambda *a, **k: None)
        sys.modules["ui_controls"] = types.SimpleNamespace(build_size_controls=lambda *a, **k: (384,384,6,0.0,False), build_batch_controls=lambda *a, **k: 4)
        sys.modules["flux_local"] = types.SimpleNamespace(set_model=lambda *a, **k: None)
        sys.modules["persistence"] = types.SimpleNamespace(dataset_rows_for_prompt=lambda p: 0, dataset_stats_for_prompt=lambda p: {"pos":0,"neg":0,"d":0}, get_dataset_for_prompt_or_session=lambda p,ss:(None,None))
        sys.modules["persistence_ui"] = types.SimpleNamespace(render_persistence_controls=lambda *a, **k: None, render_metadata_panel=lambda *a, **k: None)

        import ui_sidebar as u
        st.session_state.prompt = "opt-unique"
        st.session_state.state_path = "latent_state_test.npz"
        st.session_state.vm_choice = "XGBoost"
        lstate = types.SimpleNamespace(d=9216, w=0)
        u.render_sidebar_tail(st, lstate, st.session_state.prompt, st.session_state.state_path, st.session_state.vm_choice, 0, 0.0, "stabilityai/sd-turbo", lambda *a, **k: None, lambda *a, **k: None)

        count = sum(1 for w in writes if str(w).startswith("Optimization: Ridge only"))
        # Expect one in main and one inside the expander
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()

