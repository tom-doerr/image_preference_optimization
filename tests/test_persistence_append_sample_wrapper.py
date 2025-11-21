import os
import sys
import tempfile
import unittest
import numpy as np


class TestAppendSampleWrapper(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("IPO_DATA_ROOT", None)
        sys.modules.pop("persistence", None)

    def test_append_sample_writes_npz_and_optionally_image(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["IPO_DATA_ROOT"] = tmp.name
        import persistence

        prompt = "append-sample-wrapper"
        f = np.zeros((1, 8), dtype=float)
        # no image path
        idx1 = persistence.append_sample(prompt, f, +1.0, img=None)
        self.assertEqual(idx1, 1)
        # with image (numpy array)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        idx2 = persistence.append_sample(prompt, f, -1.0, img=img)
        self.assertEqual(idx2, 2)
        # Files on disk
        h = persistence.data_root_for_prompt(prompt)
        self.assertTrue(os.path.exists(os.path.join(h, f"{idx1:06d}", "sample.npz")))
        self.assertTrue(os.path.exists(os.path.join(h, f"{idx2:06d}", "sample.npz")))
        self.assertTrue(os.path.exists(os.path.join(h, f"{idx2:06d}", "image.png")))


if __name__ == "__main__":
    unittest.main()

