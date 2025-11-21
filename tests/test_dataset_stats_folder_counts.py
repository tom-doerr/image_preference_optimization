import os
import sys
import tempfile
import unittest
import numpy as np


class TestDatasetStatsFolderCounts(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("IPO_DATA_ROOT", None)
        sys.modules.pop("persistence", None)

    def test_counts_pos_neg_dim_and_recent(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["IPO_DATA_ROOT"] = tmp.name
        import persistence

        p = "stats-folder"
        d = 8
        f = np.zeros((1, d), dtype=float)
        # +1, -1, +1 → pos=2, neg=1, d=8, recent=[+1,-1,+1]
        persistence.append_sample(p, f, +1.0)
        persistence.append_sample(p, f, -1.0)
        persistence.append_sample(p, f, +1.0)

        s = persistence.dataset_stats_for_prompt(p)
        self.assertEqual(s.get("rows"), 3)
        self.assertEqual(s.get("pos"), 2)
        self.assertEqual(s.get("neg"), 1)
        self.assertEqual(s.get("d"), d)
        self.assertEqual(s.get("recent_labels"), [1, -1, 1])


if __name__ == "__main__":
    unittest.main()

