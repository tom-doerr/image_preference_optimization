import os
import unittest
import numpy as np


class DatasetDimMismatchResetTest(unittest.TestCase):
    def test_append_resets_on_dim_mismatch(self):
        prompt = "dim-mismatch-reset-test"
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                from persistence import dataset_path_for_prompt, dataset_rows_for_prompt, append_dataset_row

                path = dataset_path_for_prompt(prompt)
                X_old = np.zeros((2, 16384), dtype=float)
                y_old = np.array([1.0, -1.0], dtype=float)
                np.savez_compressed(path, X=X_old, y=y_old)
                self.assertEqual(dataset_rows_for_prompt(prompt), 2)

                feat_new = np.zeros((1, 12544), dtype=float)
                n = append_dataset_row(prompt, feat_new, +1.0)
                self.assertEqual(n, 1)
                self.assertEqual(dataset_rows_for_prompt(prompt), 1)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
