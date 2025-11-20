import sys
import time
import types
import unittest
import numpy as np


class SS(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


class TestAsyncXGBNonBlocking(unittest.TestCase):
    def test_xgb_fit_returns_immediately_when_async(self):
        # Small dataset with both classes
        X = np.random.randn(20, 4).astype(np.float32)
        y = np.where(np.random.rand(20) > 0.5, 1, -1).astype(np.int32)

        # Lstate stub with w/d
        class LS:
            d = 4
            w = np.zeros(4, dtype=np.float32)

        lstate = LS()

        # Session state enables async
        ss = SS()
        ss.xgb_train_async = True
        ss.ridge_train_async = True

        # Stub xgb trainer to be slow so blocking would be visible
        xv = types.ModuleType("xgb_value")

        def slow_fit(Xd, yd, n_estimators=50, max_depth=3):
            time.sleep(0.2)
            return "mdl"

        xv.fit_xgb_classifier = slow_fit
        sys.modules["xgb_value"] = xv

        import value_model as vm

        vm.fit_value_model("XGBoost", lstate, X, y, lam=1e-3, session_state=ss)

        # Non-blocking: future should be present and not done immediately
        fut = ss.get("xgb_fit_future")
        self.assertIsNotNone(fut)
        self.assertFalse(fut.done())
        # Allow it to finish to avoid leaking threads
        fut.result(timeout=2)


if __name__ == "__main__":
    unittest.main()
