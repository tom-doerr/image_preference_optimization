import os
import sys
import types
import unittest
import importlib
import numpy as np


class TestSchedulerSigmaAlignment(unittest.TestCase):
    def setUp(self):
        for m in ('torch', 'diffusers', 'flux_local'):
            sys.modules.pop(m, None)
        os.environ['FLUX_LOCAL_MODEL'] = 'dummy/model'

    def test_std_tracks_scheduler_init_sigma_after_set_timesteps(self):
        # torch stub
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
        torch.tensor = lambda x, dtype=None, device=None: Tensor(x)
        sys.modules['torch'] = torch

        recorded = {}
        # diffusers stub whose scheduler updates init_noise_sigma on set_timesteps
        diffusers = types.ModuleType('diffusers')
        class _Out:
            def __init__(self):
                self.images = ['ok']
        class _Sched:
            def __init__(self):
                self.init_noise_sigma = 1.0
                self.sigmas = types.SimpleNamespace(max=lambda: types.SimpleNamespace(item=lambda: self.init_noise_sigma))
            def set_timesteps(self, steps):
                # pretend sigma grows with steps for test purposes
                self.init_noise_sigma = 7.5 if int(steps) == 6 else 3.0
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

        lat = np.random.RandomState(0).randn(1, 4, 8, 8).astype(np.float32)
        img = flux_local.generate_flux_image_latents('p', latents=lat, width=64, height=64, steps=6, guidance=2.0)
        self.assertEqual(img, 'ok')
        L = recorded.get('latents')
        self.assertIsNotNone(L)
        s = float(L.arr.std())
        # Allow some tolerance due to float ops in mocks
        self.assertTrue(6.5 < s < 8.5, f"expected std ~7.5, got {s}")


if __name__ == '__main__':
    unittest.main()

