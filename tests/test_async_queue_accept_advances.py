import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_basic
from persistence import dataset_rows_for_prompt


class _ImmediateFuture:
    def __init__(self, value):
        self._v = value

    def done(self):
        return True

    def result(self):
        return self._v


class TestAsyncQueueAcceptAdvances(unittest.TestCase):
    def test_queue_removed(self):
        self.skipTest("Async queue removed: test skipped")


if __name__ == "__main__":
    unittest.main()
