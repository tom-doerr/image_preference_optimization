import sys
import types
import unittest
import numpy as np


class TrainToastOnStartTest(unittest.TestCase):
    def tearDown(self):
        for m in (
            "batch_ui",
            "streamlit",
            "persistence",
            "latent_logic",
            "value_model",
            "value_scorer",
            "flux_local",
        ):
            sys.modules.pop(m, None)

    def test_toast_called_when_training_starts(self):
        from tests.helpers import st_streamlit

        st, _ = st_streamlit.stub_with_writes()
        toasts = []
        st.toast = lambda msg: toasts.append(str(msg))
        st.session_state.prompt = "toast-start"
        st.session_state.steps = 3
        st.session_state.guidance_eff = 0.0
        st.session_state.batch_size = 1
        st.session_state.cur_batch = [np.zeros(4)]
        st.session_state.cur_labels = [None]
        st.session_state.lstate = types.SimpleNamespace(
            width=64,
            height=64,
            d=4,
            sigma=1.0,
            rng=np.random.default_rng(0),
            w=np.zeros(4),
        )
        st.session_state.vm_choice = "XGBoost"
        st.session_state.xgb_train_async = False
        st.session_state.reg_lambda = 1e-3

        p = types.ModuleType("persistence")
        p.get_dataset_for_prompt_or_session = lambda *a, **k: (
            np.ones((3, 4)),
            np.array([1.0, -1.0, 1.0]),
        )
        sys.modules["persistence"] = p

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        ll.sample_z_xgb_hill = lambda *a, **k: np.zeros(st.session_state.lstate.d)
        sys.modules["latent_logic"] = ll

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "ok")
        sys.modules["value_scorer"] = vs

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        sys.modules["flux_local"] = fl

        vm = types.ModuleType("value_model")
        vm.ensure_fitted = lambda *a, **k: None
        vm.fit_value_model = lambda *a, **k: None
        sys.modules["value_model"] = vm

        sys.modules["streamlit"] = st
import batch_ui

        batch_ui._curation_train_and_next()

        self.assertTrue(any("Training XGBoost" in t for t in toasts))


if __name__ == "__main__":
    unittest.main()
