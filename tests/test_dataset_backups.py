import os
import numpy as np
import unittest
from persistence import append_dataset_row


class TestDatasetBackups(unittest.TestCase):
    def test_backups_written_minutely_hourly_daily(self):
        prompt = 'backup_test_prompt'
        feat = np.zeros((1, 8))
        # Append one row; this writes a folder sample and its backup
        append_dataset_row(prompt, feat, +1.0)
        root = "."
        # Check backups folders are non-empty
        mins = os.listdir(os.path.join(root, 'backups', 'minutely'))
        hrs = os.listdir(os.path.join(root, 'backups', 'hourly'))
        days = os.listdir(os.path.join(root, 'backups', 'daily'))
        self.assertTrue(len(mins) > 0)
        self.assertTrue(len(hrs) > 0)
        self.assertTrue(len(days) > 0)


if __name__ == '__main__':
    unittest.main()
