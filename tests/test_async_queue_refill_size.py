import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class _ImmediateFuture:
    def __init__(self, value):
        self._v = value

    def done(self):
        return True

    def result(self):
        return self._v


class TestAsyncQueueRefillSize(unittest.TestCase):
    def test_queue_removed(self):
        self.skipTest("Async queue removed: test skipped")


if __name__ == "__main__":
    unittest.main()
