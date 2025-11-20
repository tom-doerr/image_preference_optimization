import unittest
import types
import numpy as np


class TestValueModelFitRecordsTime(unittest.TestCase):
    def test_fit_updates_w_and_records_time(self):
        from value_model import fit_value_model

        # simple linear separable dataset
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 4)).astype(float)
        y = np.sign(X[:, 0]).astype(float)
        lstate = types.SimpleNamespace(w=np.zeros(4, dtype=float))
        ss = {}
        fit_value_model("Ridge", lstate, X, y, lam=1e-3, session_state=ss)
        self.assertTrue(np.linalg.norm(lstate.w) > 0.0)
        self.assertIn("last_train_ms", ss)
        self.assertIn("last_train_at", ss)


if __name__ == "__main__":
    unittest.main()
