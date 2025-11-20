import sys
import types
import unittest


class TestModesDispatch(unittest.TestCase):
    def test_run_mode_calls_queue(self):
        # Minimal stub to import modes and patch target
        q = types.ModuleType("queue_ui")
        called = {"q": False}

        def rq():
            called["q"] = True

        q.run_queue_mode = rq
        sys.modules["queue_ui"] = q

        b = types.ModuleType("batch_ui")
        b.run_batch_mode = lambda: None
        sys.modules["batch_ui"] = b

        import modes

        modes.run_mode(True)
        self.assertTrue(called["q"])

    def test_run_mode_calls_batch(self):
        q = types.ModuleType("queue_ui")
        q.run_queue_mode = lambda: None
        sys.modules["queue_ui"] = q

        called = {"b": False}
        b = types.ModuleType("batch_ui")

        def rb():
            called["b"] = True

        b.run_batch_mode = rb
        sys.modules["batch_ui"] = b

        import modes

        modes.run_mode(False)
        self.assertTrue(called["b"])


if __name__ == "__main__":
    unittest.main()
