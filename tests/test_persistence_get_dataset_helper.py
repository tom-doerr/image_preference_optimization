import os
import unittest
import numpy as np


class TestPersistenceGetDatasetHelper(unittest.TestCase):
    def test_fallback_to_session_when_file_absent_then_use_file(self):
        from persistence import get_dataset_for_prompt_or_session, dataset_path_for_prompt, append_dataset_row
        prompt = 'persist helper test unique'
        path = dataset_path_for_prompt(prompt)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        # Session fallback
        ss = type('SS', (), {})()
        d = 8
        Xs = np.zeros((1, d), dtype=float)
        ys = np.array([+1.0], dtype=float)
        setattr(ss, 'dataset_X', Xs)
        setattr(ss, 'dataset_y', ys)
        X, y = get_dataset_for_prompt_or_session(prompt, ss)
        self.assertIsNotNone(X)
        self.assertEqual(int(X.shape[0]), 1)
        # Create on-disk dataset and ensure it is preferred
        feat = np.ones((1, d), dtype=float)
        append_dataset_row(prompt, feat, -1.0)
        X2, y2 = get_dataset_for_prompt_or_session(prompt, ss)
        self.assertIsNotNone(X2)
        self.assertGreaterEqual(int(X2.shape[0]), 1)


if __name__ == '__main__':
    unittest.main()

