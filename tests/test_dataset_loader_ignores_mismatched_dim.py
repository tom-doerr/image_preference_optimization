import os
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


class TestDatasetLoaderIgnoresMismatchedDim(unittest.TestCase):
    def test_loader_returns_only_matching_rows(self):
        from ipo.core.persistence import append_dataset_row, get_dataset_for_prompt_or_session, data_root_for_prompt

        prompt = "dim filter test unique"
        # Clean folder
        root = data_root_for_prompt(prompt)
        if os.path.isdir(root):
            import shutil

            shutil.rmtree(root)
        # Write one matching row (d=4) and one mismatched (d=8)
        append_dataset_row(prompt, np.zeros((1, 4), dtype=float), +1.0)
        append_dataset_row(prompt, np.zeros((1, 8), dtype=float), -1.0)

        ss = SS()
        ss.lstate = LS(d=4)

        X, y = get_dataset_for_prompt_or_session(prompt, ss)
        # Expect only the 4-dim row is included
        self.assertIsNotNone(X)
        self.assertEqual(int(X.shape[0]), 1)
        self.assertEqual(int(X.shape[1]), 4)
        self.assertEqual(int(y.shape[0]), 1)


if __name__ == "__main__":
    unittest.main()
