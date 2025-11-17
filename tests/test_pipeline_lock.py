import os
import sys
import time
import types
import threading
import unittest


class TestPipelineLock(unittest.TestCase):
    def setUp(self):
        for m in ('torch', 'diffusers', 'flux_local'):
            if m in sys.modules:
                del sys.modules[m]

    def test_calls_serialized(self):
        # Mock torch
        torch = types.ModuleType('torch')
        class _Cuda:
            @staticmethod
            def is_available():
                return True
        torch.cuda = _Cuda()
        torch.float16 = object()
        def _tensor(x, dtype=None, device=None):
            return types.SimpleNamespace(std=lambda: types.SimpleNamespace(item=lambda: 1.0),
                                         to=lambda **kw: _tensor(x, **kw),
                                         numel=lambda: 1)
        torch.tensor = _tensor
        sys.modules['torch'] = torch

        # Mock diffusers pipeline with a concurrency sentinel
        state = {'in_call': False, 'max_concurrent': 0}
        diffusers = types.ModuleType('diffusers')
        class _Sched:
            init_noise_sigma = 1.0
        class _Out:
            def __init__(self):
                self.images = ['ok']
        class _Pipe:
            scheduler = _Sched()
            def to(self, device):
                return self
            def __call__(self, **kwargs):
                # Detect overlap
                if state['in_call']:
                    state['max_concurrent'] = 2
                state['in_call'] = True
                time.sleep(0.05)
                state['in_call'] = False
                return _Out()
        class _DP:
            @staticmethod
            def from_pretrained(mid, **kwargs):
                return _Pipe()
        diffusers.DiffusionPipeline = _DP
        sys.modules['diffusers'] = diffusers

        os.environ['FLUX_LOCAL_MODEL'] = 'dummy/model'
        import flux_local

        # Call generate twice in parallel; lock should serialize to prevent overlap
        errs = []
        def worker():
            try:
                flux_local.generate_flux_image_latents('p', latents=[[[[0]]]])
            except Exception as e:
                errs.append(e)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertFalse(errs)
        self.assertNotEqual(state['max_concurrent'], 2, 'pipeline calls overlapped; lock missing')


if __name__ == '__main__':
    unittest.main()
