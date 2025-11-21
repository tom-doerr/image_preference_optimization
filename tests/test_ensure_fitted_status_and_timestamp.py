import sys
import types
import unittest
from datetime import datetime

import numpy as np

from constants import Keys


class TestEnsureFittedStatusAndTimestamp(unittest.TestCase):
    def tearDown(self):
        for m in ("value_model", "xgb_value", "latent_logic"):
            sys.modules.pop(m, None)

    def test_xgb_ensure_fitted_sets_ok_and_timestamp(self):
        # Stub xgb_value.fit_xgb_classifier (no real xgboost)
        calls = {"n": 0}

        def fit_xgb_classifier(X, y, **kwargs):
            calls["n"] = int(X.shape[0])

            class _M:
                pass

            return _M()

        xgb_mod = types.ModuleType("xgb_value")
        xgb_mod.fit_xgb_classifier = fit_xgb_classifier  # type: ignore[attr-defined]
        xgb_mod.score_xgb_proba = lambda mdl, f: 0.5
        sys.modules["xgb_value"] = xgb_mod

        from value_model import fit_value_model

        class LState:
            def __init__(self, d):
                self.d = d
                self.w = np.zeros(d, dtype=float)

        class SS(dict):
            pass

        d = 4
        X = np.vstack([np.ones((1, d)), -np.ones((1, d))]).astype(float)
        y = np.array([1.0, -1.0], dtype=float)
        lstate = LState(d)
        ss = SS()

        fit_value_model("XGBoost", lstate, X, y, lam=1e-3, session_state=ss)

        # Model cached and status ok
        cache = ss.get("xgb_cache", {})
        self.assertIsInstance(cache.get("model"), object)
        self.assertEqual(calls["n"], 2)
        st = ss.get(Keys.XGB_TRAIN_STATUS)
        self.assertIsInstance(st, dict)
        self.assertEqual(st.get("state"), "ok")
        # Timestamp recorded
        ts = ss.get(Keys.LAST_TRAIN_AT)
        self.assertIsInstance(ts, str)
        # Accept ISO-ish format; parsing may fail on very old Pythons, so be lenient
        try:
            datetime.fromisoformat(ts)
        except Exception:
            self.fail("LAST_TRAIN_AT is not ISO-8601")

    def test_ridge_ensure_fitted_sets_w_and_timestamp(self):
        # Stub ridge_fit to return ones
        ll = types.ModuleType("latent_logic")
        ll.ridge_fit = lambda X, y, lam: np.ones(X.shape[1], dtype=float)
        sys.modules["latent_logic"] = ll

        from value_model import fit_value_model

        class LState:
            def __init__(self, d):
                self.d = d
                self.w = np.zeros(d, dtype=float)

        class SS(dict):
            pass

        d = 4
        X = np.vstack([np.ones((1, d)), -np.ones((1, d))]).astype(float)
        y = np.array([1.0, -1.0], dtype=float)
        lstate = LState(d)
        ss = SS()

        fit_value_model("Ridge", lstate, X, y, lam=1e-3, session_state=ss)

        self.assertTrue(np.allclose(lstate.w, np.ones(d)))
        ts = ss.get(Keys.LAST_TRAIN_AT)
        self.assertIsInstance(ts, str)
        try:
            datetime.fromisoformat(ts)
        except Exception:
            self.fail("LAST_TRAIN_AT is not ISO-8601")


if __name__ == "__main__":
    unittest.main()
