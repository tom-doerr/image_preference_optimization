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


class TestRidgeForcedAsyncInXGB(unittest.TestCase):
    def test_ridge_runs_sync_in_xgb_mode_now(self):
        X = np.random.randn(30, 6).astype(np.float32)
        y = np.where(np.random.rand(30) > 0.5, 1, -1).astype(np.int32)

        class LS:
            d = 6
            w = np.zeros(6, dtype=np.float32)

        lstate = LS()

        ss = SS()
        ss.xgb_train_async = True

        # Slow ridge to detect blocking if run synchronously
        # Temporarily patch latent_logic.ridge_fit and restore after test
        orig_ll = sys.modules.get("latent_logic")
        ll = types.ModuleType("latent_logic")

        def slow_ridge_fit(Xd, yd, lam):
            time.sleep(0.15)
            return np.ones(Xd.shape[1], dtype=np.float32)

        ll.ridge_fit = slow_ridge_fit
        sys.modules["latent_logic"] = ll

        # Fast xgb so future exists but finishes quickly
        xv = types.ModuleType("xgb_value")
        xv.fit_xgb_classifier = lambda Xd, yd, **k: "mdl"
        sys.modules["xgb_value"] = xvfrom ipo.core import value_model as vm

        try:
            vm.fit_value_model("XGBoost", lstate, X, y, lam=1e-3, session_state=ss)
        finally:
            # Restore original module to avoid leaking stub into other tests
            if orig_ll is not None:
                sys.modules["latent_logic"] = orig_ll
            else:
                sys.modules.pop("latent_logic", None)

        # Sync now: no ridge future, weights updated to ones
        self.assertIsNone(ss.get("ridge_fit_future"))
        self.assertTrue(np.allclose(lstate.w, np.ones(6)))


if __name__ == "__main__":
    unittest.main()
