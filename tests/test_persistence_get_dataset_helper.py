import os
import unittest
import numpy as np


class TestPersistenceGetDatasetHelper(unittest.TestCase):
    def test_uses_only_persisted_file(self):
        from persistence import get_dataset_for_prompt_or_session, append_dataset_row
        import hashlib

        prompt = "persist helper test unique"
        # ensure folder clean-up handled below
        # Also clear any per-sample data folder for this prompt
        h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        root = os.path.join("data", h)
        import shutil

        if os.path.isdir(root):
            shutil.rmtree(root)
        # Without a file, helper should return (None, None) regardless of session_state
        ss = type("SS", (), {})()
        setattr(ss, "dataset_X", np.zeros((1, 8), dtype=float))
        setattr(ss, "dataset_y", np.array([+1.0], dtype=float))
        X, y = get_dataset_for_prompt_or_session(prompt, ss)
        self.assertIsNone(X)
        self.assertIsNone(y)

        # After appending to the on-disk dataset, helper should read from the file
        d = 8
        feat = np.ones((1, d), dtype=float)
        append_dataset_row(prompt, feat, -1.0)
        X2, y2 = get_dataset_for_prompt_or_session(prompt, ss)
        self.assertIsNotNone(X2)
        self.assertEqual(int(X2.shape[0]), 1)


if __name__ == "__main__":
    unittest.main()
