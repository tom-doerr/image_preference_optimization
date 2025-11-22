import os
import numpy as np
import unittest
from ipo.core.persistence import append_dataset_row


class TestDatasetBackups(unittest.TestCase):
    def test_no_legacy_backups_written(self):
        prompt = "backup_test_prompt"
        feat = np.zeros((1, 8))
        root = "."
        # Snapshot existing backup counts (if any), then verify no increase after append
        before = {}
        for sub in ("minutely", "hourly", "daily"):
            path = os.path.join(root, "backups", sub)
            before[sub] = len(os.listdir(path)) if os.path.isdir(path) else 0
        # Append one row; this writes a folder sample only (no backups)
        append_dataset_row(prompt, feat, +1.0)
        after = {}
        for sub in ("minutely", "hourly", "daily"):
            path = os.path.join(root, "backups", sub)
            after[sub] = len(os.listdir(path)) if os.path.isdir(path) else 0
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
