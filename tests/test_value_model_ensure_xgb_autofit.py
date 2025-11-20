import sys
import types
import unittest

import numpy as np


class TestEnsureFittedXGBoostAutoFit(unittest.TestCase):
    def test_ensure_fitted_trains_xgb_when_w_restored_and_cache_empty(self):
        # Stub xgb_value so we don't import real xgboost.
        calls = {"n": 0}

        def fit_xgb_classifier(X, y, **kwargs):
            calls["n"] = int(X.shape[0])

            class _M:
                pass

            return _M()

        xgb_mod = types.ModuleType("xgb_value")
        xgb_mod.fit_xgb_classifier = fit_xgb_classifier  # type: ignore[attr-defined]
        xgb_mod.score_xgb_proba = lambda mdl, f: 0.5  # pragma: no cover
        sys.modules["xgb_value"] = xgb_mod

        from value_model import ensure_fitted

        class LState:
            def __init__(self, d):
                self.d = d
                # Simulate w restored from disk: non-zero norm.
                self.w = np.ones(d, dtype=float)

        class SS(dict):
            """Dict with attribute access, like streamlit.session_state."""

            pass

        d = 4
        # Tiny separable dataset with both classes present.
        X = np.vstack([np.ones((1, d)), -np.ones((1, d))]).astype(float)
        y = np.array([1.0, -1.0], dtype=float)
        lstate = LState(d)
        ss = SS()

        # No xgb_cache and w already non-zero: ensure_fitted should still
        # trigger a single XGB fit when vm_choice == 'XGBoost'.
        ensure_fitted("XGBoost", lstate, X, y, lam=1e-3, session_state=ss)

        cache = getattr(ss, "xgb_cache", {})
        self.assertIsInstance(cache.get("model"), object)
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
