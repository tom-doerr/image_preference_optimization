import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_main_writes


class _PendingFuture:
    def done(self):
        return False

    def result(self):
        raise RuntimeError("should not be called when not done")


class TestAsyncQueueLoadingState(unittest.TestCase):
    def test_queue_removed(self):
        self.skipTest("Async queue removed: test skipped")


if __name__ == "__main__":
    unittest.main()
