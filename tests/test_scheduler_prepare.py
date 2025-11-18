import types
import unittest
import numpy as np


class DummyScheduler:
    def __init__(self):
        self._step_index = None

    def set_timesteps(self, n, device=None):
        # Simulate diffusers LCM signature; initialize step index
        self._step_index = 0


class DummyPipe:
    def __init__(self):
        self.scheduler = DummyScheduler()

    def __call__(self, **kwargs):
        # Expect that _run_pipe prepared the scheduler
        assert isinstance(self.scheduler._step_index, int)
        self.scheduler._step_index += 1
        # Return an object with an images list like diffusers
        ns = types.SimpleNamespace()
        ns.images = [np.zeros((4, 4, 3), dtype=np.uint8) + 255]
        return ns


class TestSchedulerPrepare(unittest.TestCase):
    def test_run_pipe_initializes_step_index(self):
        import importlib
        import flux_local

        importlib.reload(flux_local)
        flux_local.PIPE = DummyPipe()
        # Should not raise when scheduler._step_index starts as None
        img = flux_local._run_pipe(num_inference_steps=3)
        self.assertIsNotNone(img)


if __name__ == "__main__":
    unittest.main()

