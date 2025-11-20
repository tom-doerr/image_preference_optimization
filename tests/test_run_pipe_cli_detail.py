import io
import sys
import types
import unittest
import numpy as np
from contextlib import redirect_stdout


class RunPipeCliDetailTest(unittest.TestCase):
    def tearDown(self):
        for m in ("flux_local", "torch", "diffusers"):
            sys.modules.pop(m, None)

    def test_cli_print_includes_model_and_stats(self):
        class _Cuda:
            @staticmethod
            def is_available():
                return False

        def _tensor(arr, dtype=None, device=None):
            return np.asarray(arr)

        torch = types.SimpleNamespace(
            cuda=_Cuda(),
            float16=None,
            from_numpy=lambda a: a,
            tensor=_tensor,
        )
        sys.modules["torch"] = torch
        diff = types.ModuleType("diffusers")
        diff.DiffusionPipeline = object
        sys.modules["diffusers"] = diff

        import flux_local

        class DummySched:
            init_noise_sigma = 0.5

            def __init__(self):
                self._step_index = None

            def set_timesteps(self, n, device=None):
                self.num_inference_steps = n

        class DummyPipe:
            def __init__(self):
                self.scheduler = DummySched()

            def __call__(self, **kwargs):
                return types.SimpleNamespace(images=["ok"])

        flux_local.PIPE = DummyPipe()
        flux_local._ensure_pipe = lambda mid=None: flux_local.PIPE
        flux_local.CURRENT_MODEL_ID = "stabilityai/sd-turbo"
        flux_local.LAST_CALL["event"] = "latents_call"
        flux_local.LAST_CALL["latents_std"] = 0.25

        buf = io.StringIO()
        with redirect_stdout(buf):
            flux_local._run_pipe(
                prompt="p",
                num_inference_steps=4,
                guidance_scale=0.0,
                width=2,
                height=2,
                latents=np.zeros((1, 4, 2, 2)),
            )
        out = buf.getvalue()
        self.assertIn("model=stabilityai/sd-turbo", out)
        self.assertIn("steps=4", out)
        self.assertIn("latents_std=0.25", out)


if __name__ == "__main__":
    unittest.main()
