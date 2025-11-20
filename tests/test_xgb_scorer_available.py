import sys
import types
import unittest
import numpy as np


class XGBScorerAvailableTest(unittest.TestCase):
    def tearDown(self):
        for m in ("value_scorer", "xgb_value"):
            sys.modules.pop(m, None)

    def test_return_ok_when_model_cached(self):
        vs = types.ModuleType("value_scorer")
        # Simulate a cached scorer
        xgb_mod = types.ModuleType("xgb_value")
        def _scorer(v):
            return float(np.sum(v))
        xgb_mod.get_cached_scorer = lambda prompt, ss: _scorer
        sys.modules["xgb_value"] = xgb_mod

        # Fallback ridge scorer not needed; force ok path
        from collections import defaultdict
        ss = defaultdict(dict)

        import importlib
        sys.modules["value_scorer"] = importlib.import_module("value_scorer")

        from value_scorer import get_value_scorer_with_status

        scorer, status = get_value_scorer_with_status(
            "XGBoost",
            types.SimpleNamespace(d=4, w=np.zeros(4)),
            "p",
            ss,
        )
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(scorer(np.ones(4)), 4.0)


if __name__ == "__main__":
    unittest.main()
