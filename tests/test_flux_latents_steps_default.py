import sys
import types
import unittest
import numpy as np


class FluxLatentsStepsDefaultTest(unittest.TestCase):
    def tearDown(self):
        for m in ("flux_local", "diffusers", "torch"):
            sys.modules.pop(m, None)

    def test_steps_none_defaults_and_sets_timesteps(self):
        called = {}

        class DummySched:
            def __init__(self):
                self.set_timesteps_called = False
                self._step_index = None
                self.num_inference_steps = None

            def set_timesteps(self, n, device=None):
                called["n"] = n
                called["device"] = device
                self.set_timesteps_called = True

        class DummyPipe:
            def __init__(self):
                self.scheduler = DummySched()

            def __call__(self, **k):
                return types.SimpleNamespace(images=["ok"])

        # Stub torch so _to_cuda_fp16 works with numpy arrays
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

        fl = types.ModuleType("diffusers")
        fl.AutoencoderKL = object  # unused
        sys.modules["diffusers"] = fl

        import flux_local

        flux_local.PIPE = DummyPipe()
        flux_local._ensure_pipe = lambda mid=None: flux_local.PIPE
        out = flux_local.generate_flux_image_latents(
            "p", latents=np.zeros((1, 4, 8, 8)), width=64, height=64, steps=None, guidance=0.0
        )
        self.assertEqual(out, "ok")
        self.assertEqual(called.get("n"), 20)


if __name__ == "__main__":
    unittest.main()
