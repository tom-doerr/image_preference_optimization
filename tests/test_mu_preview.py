import unittest


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class TestMuPreview(unittest.TestCase):
    def test_mu_image_set_on_generate_pair(self):
        self.skipTest("μ preview and pair generate removed from UI")


if __name__ == "__main__":
    unittest.main()
