import unittest
import numpy as np
from ipo.core.persistence import append_dataset_row, dataset_rows_for_prompt


class TestAppendDatasetRow(unittest.TestCase):
    def test_append_returns_new_count(self):
        # Use plain helper without UI
        prompt = "append dataset row helper test"
        # Clean file
        # No NPZ to remove under folder-only scheme
        feat = np.zeros((1, 8))  # small dummy, shape only matters
        n1 = append_dataset_row(prompt, feat, +1.0)
        self.assertEqual(n1, 1)
        n2 = append_dataset_row(prompt, feat, -1.0)
        self.assertEqual(n2, 2)
        self.assertEqual(dataset_rows_for_prompt(prompt), 2)


if __name__ == "__main__":
    unittest.main()
