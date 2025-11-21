import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class DummyFuture:
    def result(self):
        return "ok-image"


class TestQueueToast(unittest.TestCase):
    def test_queue_removed(self):
        self.skipTest("Async queue removed: test skipped")


if __name__ == "__main__":
    unittest.main()
