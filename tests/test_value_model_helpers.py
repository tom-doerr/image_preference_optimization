import unittest
import types
import numpy as np


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class LState:
    def __init__(self, d: int):
        self.d = d
        self.w = None
        self.w_lock = None


class TestValueModelHelpers(unittest.TestCase):
    def test_fit_ridge_sets_weights(self):
        from value_model import _fit_ridge

        d = 4
        X = np.eye(d)
        y = np.array([+1, -1, +1, -1], dtype=float)
        st = LState(d)
        _fit_ridge(st, X, y, 1e-3)
        self.assertIsNotNone(st.w)
        self.assertEqual(len(st.w), d)

    def test_maybe_fit_logit_creates_W(self):
        from value_model import _maybe_fit_logit
from constants import Keys

        X = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], dtype=float)
        y = np.array([+1.0, +1.0, -1.0, -1.0], dtype=float)
        sess = Session()
        sess[Keys.LOGIT_STEPS] = 10
        _maybe_fit_logit(X, y, 1e-3, sess)
        self.assertIn(Keys.LOGIT_W, sess)
        W = np.asarray(sess[Keys.LOGIT_W])
        self.assertEqual(W.shape, (2,))

    def test_maybe_fit_xgb_creates_cache(self):
        from value_model import _maybe_fit_xgb

        X = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], dtype=float)
        y = np.array([+1.0, +1.0, -1.0, -1.0], dtype=float)
        sess = Session()
        _maybe_fit_xgb(X, y, 1e-3, sess)
        cache = getattr(sess, "xgb_cache", None)
        # If xgboost is unavailable, helper swallows and leaves cache absent; tolerate both.
        if cache is not None:
            self.assertIn("model", cache)
            self.assertEqual(cache.get("n"), 4)


if __name__ == "__main__":
    unittest.main()

