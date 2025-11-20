import unittest
import numpy as np
import types
import sys


class TestValueScorerZeroCases(unittest.TestCase):
    def test_distancehill_without_dataset_returns_zero_and_status(self):
        # Stub persistence.get_dataset_for_prompt_or_session to return empty,
        # but restore the real module afterwards to avoid affecting other tests.
        import persistence as real_persistence

        mod = types.ModuleType("persistence")
        mod.get_dataset_for_prompt_or_session = lambda *a, **k: (None, None)
        sys.modules["persistence"] = mod
        try:
            from value_scorer import get_value_scorer_with_status

            d = 4
            lstate = types.SimpleNamespace(d=d, w=np.ones(d, dtype=float), sigma=1.0)
            session_state = types.SimpleNamespace()
            scorer, status = get_value_scorer_with_status(
                "DistanceHill", lstate, prompt="p", session_state=session_state
            )
            f = np.array([1.0, -1.0, 0.5, 2.0])
            self.assertEqual(float(scorer(f)), 0.0)
            self.assertEqual(status, "dist_empty")
        finally:
            sys.modules["persistence"] = real_persistence

    def test_cosinehill_without_dataset_returns_zero_and_status(self):
        import persistence as real_persistence

        mod = types.ModuleType("persistence")
        mod.get_dataset_for_prompt_or_session = lambda *a, **k: (None, None)
        sys.modules["persistence"] = mod
        try:
            from value_scorer import get_value_scorer_with_status

            d = 4
            lstate = types.SimpleNamespace(d=d, w=np.ones(d, dtype=float), sigma=1.0)
            session_state = types.SimpleNamespace()
            scorer, status = get_value_scorer_with_status(
                "CosineHill", lstate, prompt="p", session_state=session_state
            )
            f = np.array([0.1, 0.2, -0.3, 0.4])
            self.assertEqual(float(scorer(f)), 0.0)
            self.assertEqual(status, "cos_empty")
        finally:
            sys.modules["persistence"] = real_persistence


if __name__ == "__main__":
    unittest.main()
