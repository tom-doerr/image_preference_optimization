import sys
import types
import unittest

from tests.helpers.st_streamlit import stub_with_writes


class TestSidebarStringsOrder(unittest.TestCase):
    def tearDown(self):
        for m in ("app", "streamlit", "flux_local", "persistence", "ui", "ui_metrics", "ui_controls", "value_scorer", "latent_opt"):
            sys.modules.pop(m, None)

    def test_canonical_train_block_order(self):
        st, writes = stub_with_writes()
        st.text_input = lambda *_, value="": "order-check"
        sys.modules["streamlit"] = st

        # Stubs to keep import light
        sys.modules["flux_local"] = types.SimpleNamespace(
            generate_flux_image=lambda *a, **k: "ok-text",
            generate_flux_image_latents=lambda *a, **k: "ok-image",
            set_model=lambda *a, **k: None,
            get_last_call=lambda: {},
        )
        sys.modules["persistence"] = types.SimpleNamespace(
            dataset_rows_for_prompt=lambda p: 0,
            dataset_stats_for_prompt=lambda p: {"pos": 0, "neg": 0, "d": 0},
            get_dataset_for_prompt_or_session=lambda p, ss: (None, None),
        )
        # No-op persistence UI
        sys.modules["persistence_ui"] = types.SimpleNamespace(
            render_persistence_controls=lambda *a, **k: None,
            render_metadata_panel=lambda *a, **k: None,
        )
        sys.modules["ui"] = types.SimpleNamespace(
            sidebar_metric_rows=lambda *a, **k: None,
            sidebar_metric=lambda *a, **k: None,
            status_panel=lambda *a, **k: None,
        )
        sys.modules["ui_metrics"] = types.SimpleNamespace(
            render_iter_step_scores=lambda *a, **k: None,
            render_mu_value_history=lambda *a, **k: None,
        )
        sys.modules["ui_controls"] = types.SimpleNamespace(
            build_size_controls=lambda *a, **k: (384, 384, 6, 0.0, False),
            build_batch_controls=lambda *a, **k: 4,
        )
        sys.modules["value_scorer"] = types.SimpleNamespace(
            get_value_scorer_with_status=lambda *a, **k: (lambda f: 0.0, "ok")
        )
        # Minimal latent_opt surface required by app wrappers
        def _init_state(width=384, height=384, d=0, seed=0):
            import numpy as _np
            h8, w8 = height // 8, width // 8
            flat = 4 * h8 * w8
            d_eff = flat
            return types.SimpleNamespace(
                width=width,
                height=height,
                d=d_eff,
                mu=_np.zeros(d_eff),
                w=_np.zeros(d_eff),
                sigma=1.0,
                rng=_np.random.default_rng(seed),
            )

        sys.modules["latent_opt"] = types.SimpleNamespace(
            init_latent_state=_init_state,
            save_state=lambda *a, **k: None,
            load_state=lambda *a, **k: _init_state(),
            dumps_state=lambda *a, **k: b"",
            loads_state=lambda *a, **k: b"",
            state_summary=lambda *a, **k: {"pairs_logged": 0, "choices_logged": 0},
        )

        # Drive ui_sidebar directly to avoid full app import complexity
        import ipo.ui.ui_sidebar as u
        # Minimal session state
        st.session_state.prompt = "order-check"
        st.session_state.state_path = "latent_state_test.npz"
        st.session_state.vm_choice = "XGBoost"
        st.session_state.iter_steps = 0
        st.session_state.iter_eta = 0.0
        lstate = types.SimpleNamespace(d=9216, w=0)
        # Call the renderer
        u.render_sidebar_tail(
            st,
            lstate,
            st.session_state.prompt,
            st.session_state.state_path,
            st.session_state.vm_choice,
            int(st.session_state.iter_steps),
            float(st.session_state.iter_eta),
            "stabilityai/sd-turbo",
            lambda *a, **k: None,
            lambda *a, **k: None,
        )

        # Collect indices of key lines (first occurrence)
        def idx(prefix):
            for i, w in enumerate(writes):
                if str(w).startswith(prefix):
                    return i
            return -1

        order = [
            "Train score:",
            "CV score:",
            "Last CV:",
            "Last train:",
            "Value scorer status:",
            "Value scorer:",
            "XGBoost active:",
            "Optimization: Ridge only",
        ]
        idxs = [idx(p) for p in order]
        # All found
        self.assertTrue(all(i >= 0 for i in idxs))
        # Monotonic increasing order
        self.assertEqual(sorted(idxs), idxs)


if __name__ == "__main__":
    unittest.main()
