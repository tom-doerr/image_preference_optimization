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


class TestAsyncRidgeNonBlocking(unittest.TestCase):
    def test_ridge_fit_runs_synchronously_now(self):
        X = np.random.randn(20, 8).astype(np.float32)
        y = np.where(np.random.rand(20) > 0.5, 1, -1).astype(np.float32)

        class LS:
            d = 8
            w = np.zeros(8, dtype=np.float32)

        lstate = LS()

        ss = SS()

        # Patch latent_logic.ridge_fit to be slow
        ll = types.ModuleType("latent_logic")

        def slow_ridge_fit(Xd, yd, lam):
            time.sleep(0.2)
            return np.ones(Xd.shape[1], dtype=np.float32)

        ll.ridge_fit = slow_ridge_fit
        sys.modules["latent_logic"] = ll
        from ipo.core import value_model as vm

        t0 = time.perf_counter()
        vm.fit_value_model("Ridge", lstate, X, y, lam=1e-3, session_state=ss)
        dt = time.perf_counter() - t0
        # Sync path: takes at least the sleep time (~0.2s)
        self.assertGreaterEqual(dt, 0.19)
        # No future is created
        self.assertIsNone(ss.get("ridge_fit_future"))
        # Weights updated
        self.assertTrue(np.allclose(lstate.w, np.ones(8)))


if __name__ == "__main__":
    unittest.main()
