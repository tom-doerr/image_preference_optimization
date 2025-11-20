import unittest
import types
import numpy as np


class TestValueScorerFallback(unittest.TestCase):
    def test_xgb_without_model_returns_zero(self):
        from value_scorer import get_value_scorer
        # lstate with w
        d = 4
        lstate = types.SimpleNamespace(d=d, w=np.ones(d, dtype=float), sigma=1.0)
        session_state = types.SimpleNamespace(xgb_cache=None)
        scorer = get_value_scorer('XGBoost', lstate, prompt='p', session_state=session_state)
        f = np.array([1.0, 2.0, 3.0, 4.0])
        # Without a cached XGB model, scorer should not silently fall back to Ridge;
        # it returns 0.0 to make the missing model visible.
        self.assertAlmostEqual(scorer(f), 0.0)


if __name__ == '__main__':
    unittest.main()
