import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic
from persistence import dataset_rows_for_prompt


class TestQueueModeSaveDataset(unittest.TestCase):
    def test_queue_removed(self):
        self.skipTest("Async queue removed: test skipped")


if __name__ == "__main__":
    unittest.main()
