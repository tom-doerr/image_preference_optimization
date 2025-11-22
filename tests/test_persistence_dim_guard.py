import unittest
import numpy as np


class SS(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


class LS:
    def __init__(self, d):
        self.d = d


class TestPersistenceDimGuard(unittest.TestCase):
    def test_mismatch_sets_flag_and_returns_none(self):
        from ipo.core.persistence import append_dataset_row, get_dataset_for_prompt_or_session

        ss = SS()
        ss.lstate = LS(d=8)
        prompt = "dim guard test"
        # Seed 1 row with different dim (4)
        f = np.random.randn(1, 4).astype(np.float32)
        append_dataset_row(prompt, f, 1.0)
        X, y = get_dataset_for_prompt_or_session(prompt, ss)
        self.assertIsNone(X)
        self.assertIsNone(y)
        self.assertIn("dataset_dim_mismatch", ss)


if __name__ == "__main__":
    unittest.main()
