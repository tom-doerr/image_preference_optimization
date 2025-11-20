import sys
import types
import unittest
import numpy as np


class TrainAsyncSingleSubmitTest(unittest.TestCase):
    def tearDown(self):
        for m in ("batch_ui", "streamlit", "persistence", "value_model", "latent_logic"):
            sys.modules.pop(m, None)

    def test_async_training_submits_once(self):
        from tests.helpers import st_streamlit

        st, _ = st_streamlit.stub_with_writes()
        st.session_state.prompt = "async-once"
        st.session_state.lstate = types.SimpleNamespace(width=64, height=64, d=4, sigma=1.0, rng=np.random.default_rng(0))
        st.session_state.reg_lambda = 0.001
        st.session_state.batch_size = 2
        st.session_state.xgb_train_async = True
        st.session_state.vm_choice = "XGBoost"

        # Minimal dataset so training is eligible
        X = np.ones((3, 4), dtype=float)
        y = np.array([1.0, -1.0, 1.0], dtype=float)

        p = types.ModuleType("persistence")
        p.get_dataset_for_prompt_or_session = lambda *a, **k: (X, y)
        sys.modules["persistence"] = p

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        ll.sample_z_xgb_hill = lambda *a, **k: np.zeros(st.session_state.lstate.d)
        sys.modules["latent_logic"] = ll

        fits = []

        vm = types.ModuleType("value_model")
        vm.ensure_fitted = lambda *a, **k: None

        def _fit_vm(choice, lstate, Xd, yd, lam, ss):
            fits.append((choice, Xd.shape[0]))

        vm.fit_value_model = _fit_vm
        sys.modules["value_model"] = vm

        sys.modules["streamlit"] = st
        import batch_ui

        st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
        st.session_state.cur_labels = [None, None]

        batch_ui._curation_train_and_next()

        self.assertEqual(len(fits), 1)
        self.assertEqual(fits[0][0], "XGBoost")
        self.assertIn("xgb_train_status", st.session_state)
        self.assertEqual(st.session_state["xgb_train_status"].get("state"), "running")


if __name__ == "__main__":
    unittest.main()
