import sys
import types
import unittest
import numpy as np


class DummyFuture:
    def __init__(self, done=False):
        self._done = done

    def done(self):
        return self._done


class SS(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


class TestXgbNoResubmitWhenFutureRunning(unittest.TestCase):
    def tearDown(self):
        for m in ("value_model", "xgb_value"):
            sys.modules.pop(m, None)

    def test_does_not_resubmit_when_future_running(self):
        # Stub xgb_value.fit_xgb_classifier to detect calls
        called = {"n": 0}

        def _fit(X, y, n_estimators=50, max_depth=3):
            called["n"] += 1
            return object()

        xv = types.ModuleType("xgb_value")
        xv.fit_xgb_classifier = _fit
        sys.modules["xgb_value"] = xv

        import value_model as vm
        from constants import Keys

        # Dataset with both classes
        X = np.random.randn(10, 4).astype(np.float32)
        y = np.array([1, -1] * 5, dtype=float)
        lstate = types.SimpleNamespace(d=4)
        ss = SS()
        ss[Keys.XGB_FIT_FUTURE] = DummyFuture(done=False)
        ss[Keys.XGB_TRAIN_ASYNC] = True

        vm.fit_value_model("XGBoost", lstate, X, y, 1e-3, ss)

        # Sync-only: fit should run immediately and ignore any stale future
        self.assertEqual(called["n"], 1)
        self.assertIn(Keys.XGB_TRAIN_STATUS, ss)
        self.assertEqual(ss[Keys.XGB_TRAIN_STATUS]["state"], "ok")
        self.assertIsNone(ss.get(Keys.XGB_FIT_FUTURE))


if __name__ == "__main__":
    unittest.main()
