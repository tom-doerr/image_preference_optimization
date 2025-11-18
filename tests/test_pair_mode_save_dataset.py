import os
import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic
from persistence import dataset_rows_for_prompt, dataset_path_for_prompt


class TestPairModeSaveDataset(unittest.TestCase):
    def test_pair_mode_removed(self):
        self.skipTest('Pair mode removed; dataset saving covered by batch tests')


if __name__ == '__main__':
    unittest.main()
