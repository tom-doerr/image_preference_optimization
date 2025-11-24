import io
import sys
import types
import unittest
import contextlib
import numpy as np


class _LState:
    def __init__(self, d=2):
        self.d = d
        self.w = np.zeros(d, dtype=float)


class _Sess(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class TrainCliPrintsTest(unittest.TestCase):
    def test_ridge_summary_prints(self):
        X = np.array([[+1.0, 0.0], [-1.0, 0.0]], dtype=float)
        y = np.array([+1.0, -1.0], dtype=float)
        lstate = _LState(d=2)
        sess = _Sess()
from value_model import fit_value_model

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fit_value_model("Ridge", lstate, X, y, lam=1.0, session_state=sess)
        out = buf.getvalue()
        self.assertIn("[train-summary] ridge", out)
        self.assertIn("rows=2", out)

    def test_logit_summary_prints(self):
        X = np.array([[+1.0, 0.0], [-1.0, 0.0], [+0.5, 0.0], [-0.5, 0.0]], dtype=float)
        y = np.array([+1.0, -1.0, +1.0, -1.0], dtype=float)
        lstate = _LState(d=2)
        sess = _Sess()
        # Make logit path active
        sess["logit_steps"] = 20
        buf = io.StringIO()
from value_model import fit_value_model

        with contextlib.redirect_stdout(buf):
            fit_value_model("Logistic", lstate, X, y, lam=0.1, session_state=sess)
        out = buf.getvalue()
        self.assertIn("[train-summary] logit", out)
        self.assertIn("rows=4", out)


if __name__ == "__main__":
    unittest.main()

