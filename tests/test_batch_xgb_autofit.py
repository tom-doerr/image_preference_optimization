import sys
import types
import unittest
import numpy as np

from tests.helpers import st_streamlit
from latent_opt import init_latent_state


class BatchXGBAutofitTest(unittest.TestCase):
    def tearDown(self):
        for name in ("streamlit", "persistence", "xgb_value", "batch_ui", "value_scorer", "latent_logic"):
            sys.modules.pop(name, None)

    def test_new_batch_autofits_xgb_when_dataset_available(self):
        # ensure clean imports
        for name in ("persistence", "value_scorer", "latent_logic", "batch_ui"):
            sys.modules.pop(name, None)
        st = st_streamlit.stub_basic()
        st.session_state.lstate = init_latent_state(width=16, height=16, seed=1)
        st.session_state.prompt = "xgb-batch-auto"
        st.session_state.vm_choice = "XGBoost"
        st.session_state.batch_size = 2
        st.session_state.iter_steps = 2
        st.session_state.lr_mu_ui = 0.3
        st.session_state.trust_r = None
        st.session_state.reg_lambda = 1e-3
        sys.modules["streamlit"] = st

        d = st.session_state.lstate.d
        X = np.vstack([np.ones((1, d)), -np.ones((1, d))])
        y = np.array([1.0, -1.0])

        persistence = types.ModuleType("persistence")
        persistence.get_dataset_for_prompt_or_session = lambda prompt, ss: (X, y)
        sys.modules["persistence"] = persistence

        xv = types.ModuleType("xgb_value")
        xv.fit_xgb_classifier = lambda X_in, y_in, **kwargs: {"trained_on": len(y_in)}
        xv.score_xgb_proba = lambda mdl, fvec: 0.8
        sys.modules["xgb_value"] = xv

        import batch_ui  # noqa: WPS433

        batch_ui._curation_new_batch()

        cache = getattr(st.session_state, "xgb_cache", {}) or {}
        self.assertIsNotNone(cache.get("model"))
        from value_scorer import get_value_scorer_with_status

        scorer, status = get_value_scorer_with_status(
            "XGBoost", st.session_state.lstate, st.session_state.prompt, st.session_state
        )
        self.assertEqual(status, "ok")
        # scorer should be callable and produce a non-zero score on the dataset.
        self.assertGreater(float(scorer(X[0])), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
