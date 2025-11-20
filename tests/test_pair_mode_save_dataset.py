import unittest


class TestPairModeSaveDataset(unittest.TestCase):
    def test_pair_mode_removed(self):
        self.skipTest('Pair mode removed; dataset saving covered by batch tests')


if __name__ == '__main__':
    unittest.main()
