import unittest
import types
import sys


class TestGuidanceTurboZero(unittest.TestCase):
    def test_turbo_guidance_forced_zero(self):
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            float16='fp16',
            tensor=lambda *a, **k: a[0],
        )
        sys.modules['torch'] = torch

        class _Pipe:
            def __init__(self):
                self.scheduler = types.SimpleNamespace(init_noise_sigma=1.0)
            def to(self, *a, **k):
                return self
            def __call__(self, **kw):
                TestGuidanceTurboZero.kw = kw
                return types.SimpleNamespace(images=[object()])
        class _DP:
            @classmethod
            def from_pretrained(cls, mid, **kw):
                return _Pipe()

        sys.modules['diffusers'] = types.SimpleNamespace(DiffusionPipeline=_DP)
        import flux_local as fl
        fl.CURRENT_MODEL_ID = "stabilityai/sd-turbo"
        fl.PIPE = None
        fl.PROMPT_CACHE.clear()
        fl._free_pipe()
        img = fl.generate_flux_image("p", width=64, height=64, steps=2, guidance=3.5)
        self.assertIsNotNone(img)
        self.assertEqual(TestGuidanceTurboZero.kw["guidance_scale"], 0.0)
        # With effective guidance 0 we skip cached prompt embeds and pass the prompt string.
        self.assertIn("prompt", TestGuidanceTurboZero.kw)
        self.assertNotIn("prompt_embeds", TestGuidanceTurboZero.kw)

    def test_turbo_guidance_zero_in_latents_path(self):
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            float16="fp16",
            tensor=lambda *a, **k: a[0],
        )
        sys.modules["torch"] = torch

        import numpy as np

        import flux_local as fl

        fl.CURRENT_MODEL_ID = "stabilityai/sd-turbo"
        fl.PIPE = types.SimpleNamespace(
            scheduler=types.SimpleNamespace(init_noise_sigma=1.0)
        )
        fl.PROMPT_CACHE.clear()

        captured = {}

        def _run_pipe(**kw):
            captured.update(kw)
            return types.SimpleNamespace(images=[object()])

        fl._run_pipe = _run_pipe  # type: ignore

        lat = np.zeros((1, 4, 8, 8), dtype=np.float32)
        img = fl.generate_flux_image_latents(
            "p", latents=lat, width=64, height=64, steps=2, guidance=5.0
        )
        self.assertIsNotNone(img)
        self.assertEqual(captured.get("guidance_scale"), 0.0)
        # When guidance is clamped to 0 we should pass the prompt string, not embeds.
        self.assertIn("prompt", captured)
        self.assertNotIn("prompt_embeds", captured)


if __name__ == "__main__":
    unittest.main()
