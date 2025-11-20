import os
import unittest
import numpy as np


class TestDataFolderSamples(unittest.TestCase):
    def test_append_creates_per_sample_folder(self):
        from persistence import append_dataset_row, get_dataset_for_prompt_or_session

        prompt = "data folder samples test"
        # Ensure clean dataset file and data folder
        # No NPZ path under folder-only scheme; ensure folder root is clean
        h = __import__("hashlib").sha1(prompt.encode("utf-8")).hexdigest()[:10]
        root = os.path.join("data", h)
        if os.path.isdir(root):
            import shutil

            shutil.rmtree(root)

        d = 4
        feat1 = np.ones((1, d), dtype=float)
        feat2 = 2 * np.ones((1, d), dtype=float)
        append_dataset_row(prompt, feat1, +1.0)
        append_dataset_row(prompt, feat2, -1.0)

        # Expect two per-sample folders with sample.npz
        s1 = os.path.join(root, "000001", "sample.npz")
        s2 = os.path.join(root, "000002", "sample.npz")
        self.assertTrue(os.path.exists(s1))
        self.assertTrue(os.path.exists(s2))
        with np.load(s1) as z1:
            X1 = z1["X"]
            y1 = z1["y"]
        with np.load(s2) as z2:
            X2 = z2["X"]
            y2 = z2["y"]
        self.assertEqual(X1.shape, (1, d))
        self.assertEqual(X2.shape, (1, d))
        self.assertAlmostEqual(float(y1[0]), 1.0)
        self.assertAlmostEqual(float(y2[0]), -1.0)

        # Loader should reconstruct X, y from data/<hash>/
        X_loaded, y_loaded = get_dataset_for_prompt_or_session(
            prompt, type("SS", (), {})()
        )
        self.assertIsNotNone(X_loaded)
        self.assertEqual(int(X_loaded.shape[0]), 2)


if __name__ == "__main__":
    unittest.main()
