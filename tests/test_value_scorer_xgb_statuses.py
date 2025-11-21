import sys
import types
import unittest
import numpy as np

from value_scorer import get_value_scorer_with_status


class TestValueScorerXGBStatuses(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("xgb_value", None)

    def test_xgb_unavailable_then_ok(self):
        # xgb_unavailable when no cache/model
        ss = {}
        l = types.SimpleNamespace(d=4)
        sc0, st0 = get_value_scorer_with_status("XGBoost", l, "p", ss)
        self.assertEqual(st0, "xgb_unavailable")
        self.assertEqual(sc0(np.ones(4)), 0.0)

        # ok when cache has a model and score fn returns a value
        def score_xgb_proba(mdl, f):
            return 0.7

        sys.modules["xgb_value"] = types.SimpleNamespace(score_xgb_proba=score_xgb_proba)
        ss = types.SimpleNamespace(xgb_cache={"model": object(), "n": 2})
        sc1, st1 = get_value_scorer_with_status("XGBoost", l, "p", ss)
        self.assertEqual(st1, "ok")
        self.assertAlmostEqual(sc1(np.zeros(4)), 0.7, places=6)

    def test_xgb_training_status(self):
        # When training is running and no model, report xgb_training
        from constants import Keys

        ss = {Keys.XGB_TRAIN_STATUS: {"state": "running"}, "xgb_cache": {}}
        l = types.SimpleNamespace(d=4)
        sc, st = get_value_scorer_with_status("XGBoost", l, "p", ss)
        self.assertEqual(st, "xgb_training")
        self.assertEqual(sc(np.ones(4)), 0.0)


if __name__ == "__main__":
    unittest.main()
