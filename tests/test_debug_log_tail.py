import sys
import types
import unittest


class TestDebugLogTail(unittest.TestCase):
    def setUp(self):
        # Write a tiny ipo.debug.log with a sentinel line
        with open("ipo.debug.log", "w", encoding="utf-8") as f:
            f.write("alpha\n")
            f.write("beta\n")
            f.write("TAIL-LINE-XYZ\n")

    def _stub_streamlit(self):
        from tests.helpers.st_streamlit import stub_with_writes

        st, writes = stub_with_writes()

        def _checkbox(label, value=False, **_):
            # Turn on the Debug checkbox only
            return True if label == "Debug" else bool(value)

        st.sidebar.checkbox = _checkbox  # type: ignore[attr-defined]
        st.sidebar.number_input = staticmethod(lambda *a, **k: 5)  # type: ignore[attr-defined]
        return st, writes

    def _install_stubs(self, st):
        # Minimal stubs for modules imported inside render_sidebar_tail
        flux_local = types.SimpleNamespace(
            set_model=lambda *a, **k: None,
            get_last_call=lambda: {
                "event": "latents_call",
                "width": 512,
                "height": 512,
                "latents_std": 0.5,
                "latents_mean": 0.0,
            },
        )
        sys.modules["flux_local"] = flux_local

        persistence_ui = types.SimpleNamespace(render_metadata_panel=lambda *a, **k: None)
        sys.modules["persistence_ui"] = persistence_ui

        latent_opt = types.SimpleNamespace(state_summary=lambda _l: {"pairs_logged": 0, "choices_logged": 0})
        sys.modules["latent_opt"] = latent_opt

        # Minimal UI shim that writes metrics to the sidebar sink
        def _smr(pairs, per_row=2):
            for k, v in pairs:
                st.sidebar.write(f"{k} {v}")

        sys.modules["ui"] = types.SimpleNamespace(sidebar_metric_rows=_smr)

        # Train results shim
        def _render_train_results_panel(st_mod, _l, _p, _vm):
            return "n/a", "n/a", "n/a", "Ridge (ok, rows=0)", "ok"

        sys.modules["ui_sidebar_train"] = types.SimpleNamespace(
            render_train_results_panel=_render_train_results_panel
        )

        # Persistence dataset stubs
        sys.modules["persistence"] = types.SimpleNamespace(
            dataset_rows_for_prompt=lambda _p: 0,
            dataset_stats_for_prompt=lambda _p: {"pos": 0, "neg": 0, "d": 0, "recent_labels": []},
            get_dataset_for_prompt_or_session=lambda _p, _s: (None, None),
        )

    def test_debug_toggle_shows_log_tail(self):
        st, writes = self._stub_streamlit()
        self._install_stubs(st)
        # Provide minimal lstate and Keys
        class L:
            d = 4

        from constants import Keys

        st.session_state[Keys.SIDEBAR_COMPACT] = True
        st.session_state[Keys.REG_LAMBDA] = 1e-3
        import ipo.ui.ui_sidebar as ui_sidebar

        ui_sidebar.render_sidebar_tail(
            st,
            L(),
            "prompt",
            "state.npz",
            "Ridge",
            3,
            0.1,
            "dummy/model",
            lambda *a, **k: None,
            lambda *a, **k: None,
        )
        # Assert sentinel from ipo.debug.log was rendered somewhere
        self.assertTrue(any("TAIL-LINE-XYZ" in w for w in writes))


if __name__ == "__main__":
    unittest.main()
