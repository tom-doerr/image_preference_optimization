import sys
import types
import unittest
import numpy as np


class LastCallLatentsStatsTest(unittest.TestCase):
    def tearDown(self):
        for m in ("flux_local", "torch", "diffusers"):
            sys.modules.pop(m, None)

    def test_last_call_records_latents_std(self):
        # Minimal stubs for torch/diffusers to let _normalize_to_init_sigma/logging run
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
        diff.DiffusionPipeline = object  # satisfy _ensure_pipe import
        sys.modules["diffusers"] = diff

        import flux_local

        class DummySched:
            def __init__(self):
                self.init_noise_sigma = 1.0
                self._step_index = None

            def set_timesteps(self, n, device=None):
                self.num_inference_steps = n

        class DummyPipe:
            def __init__(self):
                self.scheduler = DummySched()

            def __call__(self, **kwargs):
                return types.SimpleNamespace(images=[np.zeros((2, 2, 3)) + 1.0])

        flux_local.PIPE = DummyPipe()
        flux_local._ensure_pipe = lambda mid=None: flux_local.PIPE
        flux_local.CURRENT_MODEL_ID = "stabilityai/sd-turbo"
        lat = np.ones((1, 4, 2, 2), dtype=float)
        flux_local.generate_flux_image_latents("p", latents=lat, width=2, height=2, steps=3, guidance=0.0)
        lc = flux_local.get_last_call()
        self.assertAlmostEqual(lc.get("latents_std"), float(lat.std()), places=6)
        # init_sigma may be absent in stubs; ensure core stats are present
        self.assertIn("latents_std", lc)
        self.assertIn("latents_mean", lc)


if __name__ == "__main__":
    unittest.main()
