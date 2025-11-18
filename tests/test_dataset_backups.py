import os
import sys
import numpy as np
import unittest
from persistence import append_dataset_row, dataset_path_for_prompt


class TestDatasetBackups(unittest.TestCase):
    def test_backups_written_minutely_hourly_daily(self):
        prompt = 'backup_test_prompt'
        feat = np.zeros((1, 8))
        path = dataset_path_for_prompt(prompt)
        # Append one row; this writes dataset and backups
        append_dataset_row(prompt, feat, +1.0)
        base = os.path.basename(path)
        root = os.path.dirname(path) or '.'
        # Check presence of backup files (prefix match by folder)
        mins = os.listdir(os.path.join(root, 'backups', 'minutely'))
        hrs = os.listdir(os.path.join(root, 'backups', 'hourly'))
        days = os.listdir(os.path.join(root, 'backups', 'daily'))
        self.assertTrue(any(n.startswith(base + '.') and n.endswith('.npz') for n in mins))
        self.assertTrue(any(n.startswith(base + '.') and n.endswith('.npz') for n in hrs))
        self.assertTrue(any(n.startswith(base + '.') and n.endswith('.npz') for n in days))


if __name__ == '__main__':
    unittest.main()

