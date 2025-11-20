import sys
import types
import unittest
import numpy as np


class GuidanceTurboClampTest(unittest.TestCase):
    def tearDown(self):
        for m in ("flux_local", "torch", "diffusers"):
            sys.modules.pop(m, None)

    def test_guidance_clamped_in_last_call_for_turbo(self):
        # Minimal torch/diffusers stubs
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
            def __init__(self):
                self._step_index = None

            def set_timesteps(self, n, device=None):
                self.num_inference_steps = n

        class DummyPipe:
            def __init__(self):
                self.scheduler = DummySched()

            def __call__(self, **kwargs):
                return types.SimpleNamespace(images=[np.zeros((2, 2, 3)) + 0.5])

        flux_local.PIPE = DummyPipe()
        flux_local._ensure_pipe = lambda mid=None: flux_local.PIPE
        flux_local.CURRENT_MODEL_ID = "stabilityai/sd-turbo"

        flux_local.generate_flux_image_latents(
            "p",
            latents=np.zeros((1, 4, 2, 2)),
            width=2,
            height=2,
            steps=3,
            guidance=3.5,
        )
        lc = flux_local.get_last_call()
        self.assertAlmostEqual(lc.get("guidance"), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
