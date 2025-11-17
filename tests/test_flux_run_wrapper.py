import unittest


class TestFluxRunWrapper(unittest.TestCase):
    def test_generate_text_uses_run_pipe(self):
        import types
        import sys
        # Stub out heavy imports before importing flux_local
        torch = types.ModuleType('torch')
        class _Cuda:
            @staticmethod
            def is_available():
                return True
        torch.cuda = _Cuda()
        # minimal dtype attr for code path
        torch.float16 = object()
        sys.modules['torch'] = torch
        diffusers = types.ModuleType('diffusers')
        class _DP:
            @staticmethod
            def from_pretrained(*a, **k):
                class _P:
                    def to(self, *a, **k):
                        return self
                return _P()
        diffusers.DiffusionPipeline = _DP
        sys.modules['diffusers'] = diffusers
        import flux_local as fl
        import os
        os.environ['FLUX_LOCAL_MODEL'] = 'dummy/model'
        calls = {}
        fl._run_pipe = lambda **kw: calls.setdefault('kw', kw)
        out = fl.generate_flux_image('p', steps=5, guidance=2.0, width=320, height=256)
        self.assertIsInstance(out, dict)
        self.assertIn('prompt', calls['kw'])
        self.assertNotIn('latents', calls['kw'])

    def test_generate_latents_uses_run_pipe_with_latents(self):
        import types
        import sys
        torch = types.ModuleType('torch')
        class _Cuda:
            @staticmethod
            def is_available():
                return True
        torch.cuda = _Cuda()
        torch.float16 = object()
        sys.modules['torch'] = torch
        diffusers = types.ModuleType('diffusers')
        class _DP:
            @staticmethod
            def from_pretrained(*a, **k):
                class _P:
                    def to(self, *a, **k):
                        return self
                return _P()
        diffusers.DiffusionPipeline = _DP
        sys.modules['diffusers'] = diffusers
        import flux_local as fl
        import os
        os.environ['FLUX_LOCAL_MODEL'] = 'dummy/model'
        # Avoid torch deps in test by stubbing converters
        fl._to_cuda_fp16 = lambda x: x
        fl._normalize_to_init_sigma = lambda pipe, x: x
        calls = {}
        fl._run_pipe = lambda **kw: calls.setdefault('kw', kw)
        out = fl.generate_flux_image_latents('p', latents='LAT', steps=4, guidance=1.5, width=64, height=64)
        self.assertIsInstance(out, dict)
        self.assertIn('latents', calls['kw'])
        self.assertEqual(calls['kw']['latents'], 'LAT')


if __name__ == '__main__':
    unittest.main()
