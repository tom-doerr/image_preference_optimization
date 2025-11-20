import sys
import types
import unittest
import numpy as np


class SS(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


class LS:
    def __init__(self, d=4):
        self.d = d
        self.w = np.zeros(d, dtype=np.float32)
        self.sigma = 1.0
        # minimal fields used by z_from_prompt
        self.width = 32
        self.height = 32
        self.rng = np.random.default_rng(0)


class TestXGBStatusTransition(unittest.TestCase):
    def test_status_goes_from_unavailable_or_training_to_ok(self):
        prompt = "status transition test"
        lstate = LS(d=4)
        ss = SS()
        ss.xgb_train_async = True

        # Seed a tiny dataset in folder storage so trainer has data
        from persistence import append_dataset_row

        for sgn in (+1, -1, +1, -1):
            f = (np.random.randn(1, lstate.d)).astype(np.float32)
            append_dataset_row(prompt, f, float(sgn))

        # Stub XGB components
        xv = types.ModuleType("xgb_value")
        xv._mdl = object()

        def fit_xgb_classifier(X, y, n_estimators=50, max_depth=3):
            return xv._mdl

        def score_xgb_proba(mdl, fvec):
            return float(np.tanh(np.sum(fvec)))

        xv.fit_xgb_classifier = fit_xgb_classifier
        xv.score_xgb_proba = score_xgb_proba
        sys.modules["xgb_value"] = xv

        import value_model as vm
        from value_scorer import get_value_scorer_with_status

        # Kick off async training
        X = np.random.randn(8, lstate.d).astype(np.float32)
        y = np.where(np.random.rand(8) > 0.5, 1, -1).astype(np.int32)
        vm.fit_value_model("XGBoost", lstate, X, y, lam=1e-3, session_state=ss)

        # Before completion: expect xgb_training or xgb_unavailable
        _, status0 = get_value_scorer_with_status("XGBoost", lstate, prompt, ss)
        self.assertIn(status0, ("xgb_training", "xgb_unavailable"))

        # Wait for background to finish
        fut = ss.get("xgb_fit_future")
        self.assertIsNotNone(fut)
        fut.result(timeout=2)

        # After completion: scorer should report ok
        _, status1 = get_value_scorer_with_status("XGBoost", lstate, prompt, ss)
        self.assertEqual(status1, "ok")


if __name__ == "__main__":
    unittest.main()
