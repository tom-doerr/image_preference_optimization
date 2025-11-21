import sys
import types
import unittest
import numpy as np

from value_scorer import get_value_scorer_with_status


class TestValueScorerRidgeStatus(unittest.TestCase):
    def test_ridge_untrained_and_ok(self):
        class LState:
            def __init__(self, d):
                self.d = d
                self.w = np.zeros(d, dtype=float)

        l0 = LState(4)
        sc0, st0 = get_value_scorer_with_status("Ridge", l0, "p", {})
        self.assertEqual(st0, "ridge_untrained")
        self.assertEqual(sc0(np.ones(4)), 0.0)

        l1 = LState(4)
        l1.w[:] = 1.0
        sc1, st1 = get_value_scorer_with_status("Ridge", l1, "p", {})
        self.assertEqual(st1, "ok")
        self.assertAlmostEqual(sc1(np.array([1.0, 0.0, 0.0, 0.0])), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

