import threading
import time
import sys
import types
import unittest


class _Sched:
    def __init__(self):
        self.num_inference_steps = None
        self._step_index = None

    def set_timesteps(self, n, device=None):
        self.num_inference_steps = int(n)
        self._step_index = 0


class _Pipe:
    def __init__(self):
        self.scheduler = _Sched()

    def __call__(self, **k):
        # Must be prepared before call
        assert self.scheduler.num_inference_steps is not None
        return types.SimpleNamespace(images=["ok"])  # like diffusers output


class TestSetModelNoRace(unittest.TestCase):
    def tearDown(self):
        for m in ("flux_local",):
            sys.modules.pop(m, None)

    def test_set_model_guarded_by_lock(self):
        import flux_local

        # Inject dummy pipe and current model
        flux_local.PIPE = _Pipe()
        flux_local.CURRENT_MODEL_ID = "id0"

        # Thread that attempts to set_model while we run the pipe
        def _t():
            time.sleep(0.01)
            # Simulate a scheduler swap by replacing PIPE.scheduler
            flux_local.set_model("id0")

        th = threading.Thread(target=_t)
        th.start()

        # Call run_pipe; with locking, this should succeed and return 'ok'
        out = flux_local._run_pipe(
            prompt="p", num_inference_steps=6, guidance_scale=0.0, width=64, height=64
        )
        self.assertEqual(out, "ok")
        th.join()


if __name__ == "__main__":
    unittest.main()

