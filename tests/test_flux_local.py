import os
import sys
import types
import unittest
import importlib


class TestFluxLocal(unittest.TestCase):
    def setUp(self):
        for m in ('torch', 'diffusers'):
            if m in sys.modules:
                del sys.modules[m]

    def test_raises_when_requirements_missing(self):
        os.environ.pop("FLUX_LOCAL_MODEL", None)
        import flux_local
        with self.assertRaises(ValueError):
            flux_local.generate_flux_image("test")

    def test_success_path_with_mocks(self):
        os.environ["FLUX_LOCAL_MODEL"] = "dummy/model"

        # Fake torch with CUDA available
        torch = types.ModuleType('torch')
        class _Cuda:
            @staticmethod
            def is_available():
                return True
        class _Gen:
            def __init__(self, device=None):
                pass
            def manual_seed(self, x):
                return self
        torch.cuda = _Cuda()
        torch.float16 = object()
        torch.Generator = _Gen
        def _tensor(x, dtype=None, device=None):
            t = types.SimpleNamespace()
            t.device = device
            return t
        torch.tensor = _tensor
        sys.modules['torch'] = torch

        # Fake diffusers pipeline
        diffusers = types.ModuleType('diffusers')
        class _Out:
            def __init__(self):
                self.images = ["ok-image"]
        class _Pipe:
            def to(self, device):
                return self
            def __call__(self, **kwargs):
                return _Out()
        class _DP:
            @staticmethod
            def from_pretrained(mid, torch_dtype=None, **kwargs):
                return _Pipe()
        diffusers.DiffusionPipeline = _DP
        sys.modules['diffusers'] = diffusers

        import flux_local
        importlib.reload(flux_local)
        latents = [[[ [0.0 for _ in range(320//8)] for _ in range(256//8) ] for _ in range(4)]]
        img = flux_local.generate_flux_image_latents("prompt", latents=latents, width=320, height=256, steps=5, guidance=2.0)
        self.assertEqual(img, "ok-image")


if __name__ == '__main__':
    unittest.main()
