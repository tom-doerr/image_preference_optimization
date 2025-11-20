import unittest


class TestBackgroundWorkers(unittest.TestCase):
    def test_decode_executor_two_workers(self):
        import background

        background.reset_executor()
        ex = background.get_executor()
        # CPython ThreadPoolExecutor exposes _max_workers; use getattr to avoid failure on alt impls
        n = getattr(ex, "_max_workers", None)
        self.assertEqual(n, 2, msg=f"expected 2 decode workers, got {n}")


if __name__ == "__main__":
    unittest.main()
