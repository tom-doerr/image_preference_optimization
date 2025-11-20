import sys
import types
import unittest
import numpy as np


class DummySched:
    def __init__(self):
        self._step_index = None
        self.num_inference_steps = None
        self.calls = []

    def set_timesteps(self, n, device=None):
        self.calls.append((n, device))
        self.num_inference_steps = n


class DummyPipe:
    def __init__(self):
        self.scheduler = DummySched()

    def __call__(self, **kwargs):
        return types.SimpleNamespace(images=["ok"])


class RunPipeSetsTimestepsTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("flux_local", None)

    def test_run_pipe_sets_timesteps_when_none(self):
        import flux_local

        pipe = DummyPipe()
        flux_local.PIPE = pipe
        out = flux_local._run_pipe(
            prompt="p",
            num_inference_steps=7,
            guidance_scale=0.0,
            width=1,
            height=1,
            latents=np.zeros((1, 4, 2, 2)),
        )
        self.assertEqual(out, "ok")
        self.assertEqual(pipe.scheduler.num_inference_steps, 7)
        self.assertEqual(pipe.scheduler._step_index, 0)
        self.assertTrue(pipe.scheduler.calls)
        n, device = pipe.scheduler.calls[0]
        self.assertEqual(n, 7)
        # device may be None under CPU test stubs; just ensure call recorded


if __name__ == "__main__":
    unittest.main()
