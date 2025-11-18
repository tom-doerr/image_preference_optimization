import time
import unittest

import background as bg


class DummyFuture:
    def __init__(self, done=False):
        self._done = done
        self._value = 'ok'

    def done(self):
        return self._done

    def result(self):
        return self._value


class TestBackgroundResultOrSync(unittest.TestCase):
    def test_returns_future_result_when_done(self):
        f = DummyFuture(done=True)
        res, out_f = bg.result_or_sync_after(f, time.time(), 1.0, lambda: 'sync')
        self.assertEqual(res, 'ok')
        self.assertIs(out_f, f)

    def test_calls_sync_after_timeout(self):
        # Not-done future and elapsed > timeout → call sync
        f = DummyFuture(done=False)
        called = {}
        def sync():
            called['x'] = True
            return 'sync'
        res, out_f = bg.result_or_sync_after(f, time.time() - 10.0, 0.1, sync)
        self.assertEqual(res, 'sync')
        self.assertTrue(called.get('x'))


if __name__ == '__main__':
    unittest.main()

