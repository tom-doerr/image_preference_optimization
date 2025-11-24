import io
import sys
import types
import unittest

import numpy as np


class TestXGBLogging(unittest.TestCase):
    def test_fit_value_model_logs_xgb_training(self):
        # Stub xgb_value.fit_xgb_classifier to avoid real xgboost import.
        calls = {"n": 0}

        def fit_xgb_classifier(X, y, **kwargs):
            calls["n"] = int(X.shape[0])

            class _M:
                pass

            return _M()

        xgb_mod = types.ModuleType("xgb_value")
        xgb_mod.fit_xgb_classifier = fit_xgb_classifier  # type: ignore[attr-defined]
        xgb_mod.score_xgb_proba = lambda mdl, f: 0.5  # unused here
        sys.modules["xgb_value"] = xgb_mod
from value_model import fit_value_model

        class LState:
            def __init__(self, d):
                self.d = d
                self.w = np.zeros(d, dtype=float)

        class SS(dict):
            pass

        d = 4
        X = np.vstack([np.ones((1, d)), -np.ones((1, d))])
        y = np.array([1.0, -1.0], dtype=float)
        lstate = LState(d)
        ss = SS()

        buf = io.StringIO()
        old_out = sys.stdout
        try:
            sys.stdout = buf
            fit_value_model("XGBoost", lstate, X, y, lam=1e-3, session_state=ss)
        finally:
            sys.stdout = old_out
        out = buf.getvalue()
        self.assertIn("[xgb] train start", out)
        self.assertIn("[xgb] train done", out)
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
