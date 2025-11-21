import unittest


class TestBackgroundWorkers(unittest.TestCase):
    def test_background_module_removed(self):
        try:
            __import__("background")
            present = True
        except Exception:
            present = False
        self.assertFalse(present)


if __name__ == "__main__":
    unittest.main()
