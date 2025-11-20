import sys
import types
import unittest
import numpy as np


class XGBScorerModelCacheTest(unittest.TestCase):
    def tearDown(self):
        for m in ("value_scorer", "xgb_value"):
            sys.modules.pop(m, None)

    def test_model_in_cache_returns_ok(self):
        # Stub xgb_value.score_xgb_proba to use cached model
        xv = types.ModuleType("xgb_value")
        xv.score_xgb_proba = lambda mdl, vec: float(np.sum(vec)) + getattr(mdl, "bias", 0.0)
        sys.modules["xgb_value"] = xv

        import value_scorer

        ss = types.SimpleNamespace(xgb_cache={"model": types.SimpleNamespace(bias=1.0)}, xgb_fit_future=None)
        lstate = types.SimpleNamespace(d=3, w=np.zeros(3))
        scorer, status = value_scorer.get_value_scorer_with_status("XGBoost", lstate, "p", ss)
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(scorer(np.array([1.0, 2.0, 3.0])), 7.0)


if __name__ == "__main__":
    unittest.main()
