import sys
import types
import unittest


class _DummySched:
    def __init__(self):
        self.set_called = False
        self._step_index = None
        self.num_inference_steps = None

    def set_timesteps(self, n, device=None):
        # Must be called before the PIPE __call__ proceeds
        self.set_called = True
        self.num_inference_steps = int(n)


class _DummyPipe:
    def __init__(self):
        self.scheduler = _DummySched()

    def __call__(self, **kwargs):
        # Ensure scheduler was prepared
        assert self.scheduler.set_called
        assert isinstance(self.scheduler.num_inference_steps, int)
        return types.SimpleNamespace(images=["ok"])  # like diffusers Output


class TestSchedulerPrepareUnderLock(unittest.TestCase):
    def tearDown(self):
        for m in ("flux_local",):
            sys.modules.pop(m, None)

    def test_prepare_and_call_ok(self):
        import flux_local

        # Inject dummy PIPE and model id
        flux_local.PIPE = _DummyPipe()
        flux_local.CURRENT_MODEL_ID = "dummy/model"

        out = flux_local._run_pipe(
            prompt="p",
            num_inference_steps=6,
            guidance_scale=0.0,
            width=64,
            height=64,
        )
        self.assertEqual(out, "ok")


if __name__ == "__main__":
    unittest.main()

