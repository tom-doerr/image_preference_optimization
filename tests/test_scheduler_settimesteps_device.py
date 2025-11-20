import types
import unittest


class DummyScheduler:
    def __init__(self):
        self._calls = []
        self._step_index = None

    def set_timesteps(self, n, device=None):
        # Record args and initialize index
        self._calls.append((int(n), device))
        self._step_index = 0


class DummyPipe:
    def __init__(self):
        self.scheduler = DummyScheduler()

    def __call__(self, **kwargs):
        ns = types.SimpleNamespace()
        ns.images = [None]
        return ns


class TestSchedulerSetTimestepsDevice(unittest.TestCase):
    def test_run_pipe_passes_cuda_device(self):
        import importlib
        import flux_local

        importlib.reload(flux_local)
        flux_local.PIPE = DummyPipe()
        try:
            flux_local._run_pipe(num_inference_steps=7)
        except Exception:
            # The dummy returns None image; we care about scheduler call
            pass
        calls = flux_local.PIPE.scheduler._calls
        self.assertTrue(any(dev == "cuda" for (_n, dev) in calls))


if __name__ == "__main__":
    unittest.main()
