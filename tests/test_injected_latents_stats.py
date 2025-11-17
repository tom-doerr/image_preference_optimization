import os
import sys
import types
import unittest
import importlib
import numpy as np


class TestInjectedLatentsStats(unittest.TestCase):
    def setUp(self):
        for m in ('torch', 'diffusers', 'flux_local'):
            if m in sys.modules:
                del sys.modules[m]

    def test_pipeline_receives_zero_mean_latents(self):
        os.environ['FLUX_LOCAL_MODEL'] = 'dummy/model'

        # Minimal torch mock with Tensor semantics used by flux_local
        torch = types.ModuleType('torch')
        class _Cuda:
            @staticmethod
            def is_available():
                return True
        torch.cuda = _Cuda()
        torch.float16 = object()

        class Tensor:
            def __init__(self, arr):
                self.arr = np.asarray(arr, dtype=np.float32)
            def to(self, device=None, dtype=None):
                return self
            def std(self):
                v = float(self.arr.std())
                return types.SimpleNamespace(item=lambda: v)
            def numel(self):
                return int(self.arr.size)

        def _tensor(x, dtype=None, device=None):
            return Tensor(x)
        torch.tensor = _tensor
        sys.modules['torch'] = torch

        # Diffusers mock that records latents and exposes a simple scheduler
        recorded = {}
        diffusers = types.ModuleType('diffusers')
        class _Out:
            def __init__(self):
                self.images = ['ok']
        class _Sched:
            init_noise_sigma = 1.0
        class _Pipe:
            def __init__(self):
                self.scheduler = _Sched()
            def to(self, device):
                return self
            def __call__(self, **kwargs):
                recorded.update(kwargs)
                return _Out()
        class _DP:
            @staticmethod
            def from_pretrained(mid, torch_dtype=None, **kwargs):
                return _Pipe()
        diffusers.DiffusionPipeline = _DP
        sys.modules['diffusers'] = diffusers

        import flux_local
        importlib.reload(flux_local)

        lat = np.random.RandomState(0).randn(1,4,32,40).astype(np.float32)
        img = flux_local.generate_flux_image_latents('p', latents=lat, width=320, height=256, steps=5, guidance=2.0)
        self.assertEqual(img, 'ok')
        L = recorded.get('latents')
        # Global mean should be near 0 after our standardization; std near init sigma (=1)
        self.assertIsNotNone(L)
        m = float(L.arr.mean())
        s = float(L.arr.std())
        # Allow small drift due to float16 conversions in mocks
        self.assertLess(abs(m), 5e-2)
        self.assertTrue(0.7 < s < 1.3)


if __name__ == '__main__':
    unittest.main()
