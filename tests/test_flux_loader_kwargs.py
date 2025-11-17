import sys
import types
import unittest
import importlib


class TestFluxLoaderKwargs(unittest.TestCase):
    def setUp(self):
        for m in ('torch', 'diffusers', 'flux_local'):
            if m in sys.modules:
                del sys.modules[m]

    def _install_mocks(self):
        # Fake torch with CUDA available
        torch = types.ModuleType('torch')
        class _Cuda:
            @staticmethod
            def is_available():
                return True
        torch.cuda = _Cuda()
        torch.float16 = object()
        sys.modules['torch'] = torch

        # Fake diffusers pipeline
        calls = {'count': 0, 'kwargs': []}
        diffusers = types.ModuleType('diffusers')
        class _Pipe:
            def __init__(self):
                self.to_called = False
            def to(self, device):
                assert device == 'cuda'
                self.to_called = True
                return self
        class _DP:
            @staticmethod
            def from_pretrained(mid, **kwargs):
                calls['count'] += 1
                calls['kwargs'].append(kwargs)
                return _Pipe()
        diffusers.DiffusionPipeline = _DP
        sys.modules['diffusers'] = diffusers
        return calls

    def test_loader_kwargs_and_caching(self):
        calls = self._install_mocks()
        import flux_local
        importlib.reload(flux_local)
        flux_local.set_model('id1')
        self.assertEqual(calls['count'], 1)
        self.assertTrue(any('low_cpu_mem_usage' in k for k in calls['kwargs']))
        self.assertTrue(any(k.get('low_cpu_mem_usage') is False for k in calls['kwargs']))
        # No device_map should be passed when using .to("cuda")
        self.assertTrue(all('device_map' not in k for k in calls['kwargs']))

        # Calling set_model with same id should not reload
        flux_local.set_model('id1')
        self.assertEqual(calls['count'], 1)

        # Changing id should reload
        flux_local.set_model('id2')
        self.assertEqual(calls['count'], 2)


if __name__ == '__main__':
    unittest.main()
