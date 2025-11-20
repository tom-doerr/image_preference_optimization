import os
import unittest
import numpy as np


class DatasetDimMismatchResetTest(unittest.TestCase):
    def test_append_counts_by_dim_and_all(self):
        prompt = "dim-mismatch-reset-test"
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                from persistence import dataset_rows_for_prompt, append_dataset_row, dataset_rows_for_prompt_dim

                # Seed two rows via folder-based append (dim 16384)
                append_dataset_row(prompt, np.zeros((1, 16384), dtype=float), +1.0)
                append_dataset_row(prompt, np.zeros((1, 16384), dtype=float), -1.0)
                self.assertEqual(dataset_rows_for_prompt(prompt), 2)

                feat_new = np.zeros((1, 12544), dtype=float)
                n = append_dataset_row(prompt, feat_new, +1.0)
                self.assertGreaterEqual(n, 1)
                # Total now sums both dims: 2 old + 1 new
                self.assertEqual(dataset_rows_for_prompt(prompt), 3)
                # Per-dim count for the new dim is 1
                self.assertEqual(dataset_rows_for_prompt_dim(prompt, 12544), 1)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
