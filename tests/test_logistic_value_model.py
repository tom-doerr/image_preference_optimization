import types
import unittest
import numpy as np


class _LState:
    def __init__(self, d=2):
        self.d = d
        self.w = np.zeros(d, dtype=float)


class _Sess(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class TestLogisticValueModel(unittest.TestCase):
    def test_logistic_unavailable_initially(self):
        from value_scorer import get_value_scorer

        lstate = _LState(d=2)
        sess = _Sess()
        scorer, tag = get_value_scorer("Logistic", lstate, "p", sess)
        self.assertIsNone(scorer)
        self.assertIn("logit", str(tag))

    def test_logistic_fit_and_predict(self):
        # Tiny linearly separable dataset in 2D
        X = np.array([[+1.0, 0.0], [-1.0, 0.0], [+0.5, 0.0], [-0.5, 0.0]], dtype=float)
        y = np.array([+1.0, -1.0, +1.0, -1.0], dtype=float)
        lstate = _LState(d=2)
        sess = _Sess()
        from value_model import fit_value_model
        fit_value_model("Logistic", lstate, X, y, lam=1.0, session_state=sess)
        from value_scorer import get_value_scorer
        scorer, tag = get_value_scorer("Logistic", lstate, "p", sess)
        self.assertIsNotNone(scorer)
        self.assertTrue(callable(scorer))
        # Positive example should have higher probability
        p_pos = float(scorer(X[0]))
        p_neg = float(scorer(X[1]))
        self.assertGreater(p_pos, 0.5)
        self.assertLess(p_neg, 0.5)


if __name__ == "__main__":
    unittest.main()

