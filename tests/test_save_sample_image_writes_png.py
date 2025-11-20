import os
import sys
import types
import unittest
import numpy as np


class SaveSampleImageTest(unittest.TestCase):
    def tearDown(self):
        for m in ("persistence",):
            sys.modules.pop(m, None)
        import shutil
        if os.path.isdir("data"):
            shutil.rmtree("data")

    def test_save_sample_image_creates_png(self):
        import persistence
        from PIL import Image

        prompt = "image-save-test"
        feat = np.zeros((1, 4), dtype=float)
        row_idx = persistence.append_dataset_row(prompt, feat, +1.0)
        img = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
        persistence.save_sample_image(prompt, row_idx, img)
        h = __import__("hashlib").sha1(prompt.encode("utf-8")).hexdigest()[:10]
        path = os.path.join("data", h, f"{row_idx:06d}", "image.png")
        self.assertTrue(os.path.exists(path))
        with Image.open(path) as im:
            self.assertEqual(im.size, (4, 4))


if __name__ == "__main__":
    unittest.main()
