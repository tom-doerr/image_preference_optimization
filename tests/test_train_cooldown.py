import sys
import types
import unittest
from datetime import datetime, timezone, timedelta

import numpy as np
from tests.helpers.st_streamlit import stub_basic


class TestTrainCooldown(unittest.TestCase):
    def tearDown(self):
        for m in (
            "batch_ui",
            "streamlit",
            "persistence",
            "latent_logic",
            "value_model",
            "flux_local",
            "value_scorer",
        ):
            sys.modules.pop(m, None)

    def test_cooldown_skips_fit_when_recent(self):
        st = stub_basic()
        st.session_state.prompt = "cooldown-test"
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
        # Set a recent train timestamp inside the cooldown window
        st.session_state.last_train_at = (
            datetime.now(timezone.utc) - timedelta(seconds=2)
        ).isoformat(timespec="seconds")
        st.session_state.min_train_interval_s = 10.0
        st.session_state.vm_choice = "XGBoost"
        st.session_state.xgb_train_async = True

        # Stubs
        p = types.ModuleType("persistence")
        p.get_dataset_for_prompt_or_session = lambda prompt, ss: (
            np.zeros((2, st.session_state.lstate.d)),
            np.array([1.0, -1.0]),
        )
        sys.modules["persistence"] = p

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        ll.z_to_latents = lambda lstate, z: np.zeros((1, 1, 2, 2))
        ll.sample_z_xgb_hill = lambda *a, **k: np.zeros(st.session_state.lstate.d)
        sys.modules["latent_logic"] = ll

        vm = types.SimpleNamespace()
        vm.fit_calls = []

        def fit_value_model(vm_choice, lstate, X, y, lam, session_state):
            vm.fit_calls.append((vm_choice, X.shape[0]))

        def ensure_fitted(vm_choice, lstate, X, y, lam, session_state):
            return None

        vm_module = types.ModuleType("value_model")
        vm_module.fit_value_model = fit_value_model
        vm_module.ensure_fitted = ensure_fitted
        sys.modules["value_model"] = vm_module

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "ok")
        sys.modules["value_scorer"] = vs

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        sys.modules["flux_local"] = fl

        sys.modules["streamlit"] = st
import batch_ui

        batch_ui._curation_train_and_next()

        self.assertEqual(
            len(vm.fit_calls), 0, "fit_value_model should be skipped during cooldown"
        )


if __name__ == "__main__":
    unittest.main()
