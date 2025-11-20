import sys
import types
import unittest


class TestFluxLocalMetaToFix(unittest.TestCase):
    def test_meta_to_reload_path(self):
        # Stub torch+cuda available
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            float16='fp16',
        )
        sys.modules['torch'] = torch

        # Create a dummy pipeline object whose .to('cuda') raises NotImplementedError
        class _P:
            def __init__(self):
                self.safety_checker = None
                self.feature_extractor = None
                self.scheduler = types.SimpleNamespace(init_noise_sigma=1.0)
            def to(self, *a, **k):
                raise NotImplementedError("Cannot copy out of meta tensor; no data!")

        # Second path: from_pretrained called with device_map='cuda' returns a working obj
        class _P2(_P):
            pass

        class _DP:
            calls = []
            @classmethod
            def from_pretrained(cls, mid, **kw):
                cls.calls.append(kw)
                if kw.get('device_map') == 'cuda':
                    return _P2()
                return _P()

        diffusers = types.SimpleNamespace(DiffusionPipeline=_DP)
        sys.modules['diffusers'] = diffusers

        # Import flux_local and call _ensure_pipe; it should reload with device_map='cuda'
        import flux_local as fl
        fl.CURRENT_MODEL_ID = None
        def _get_model_id():
            return 'dummy/model'
        fl._get_model_id = _get_model_id  # type: ignore
        fl._free_pipe()
        fl._ensure_pipe(None)
        # Verify that the second call had a device_map targeting cuda
        def _is_cuda(dm):
            return (dm == 'cuda') or (isinstance(dm, dict) and dm.get('') == 'cuda')

        self.assertTrue(any(_is_cuda(k.get('device_map')) for k in _DP.calls))
        # And low_cpu_mem_usage=True when device_map is used (diffusers requirement)
        dev_calls = [k for k in _DP.calls if _is_cuda(k.get('device_map'))]
        self.assertTrue(all(k.get('low_cpu_mem_usage') is True for k in dev_calls))

    def test_meta_to_then_meta_again_falls_back_to_cpu_reload(self):
        """If both .to() and device_map reload hit meta tensors, we retry CPU load."""
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            float16='fp16',
        )
        sys.modules['torch'] = torch

        class _P:
            def __init__(self):
                self.safety_checker = None
                self.feature_extractor = None
                self.scheduler = types.SimpleNamespace(init_noise_sigma=1.0)
            def to(self, *a, **k):
                raise NotImplementedError("meta again")

        class _P3(_P):
            def to(self, *a, **k):
                return self

        class _DP:
            calls = []
            @classmethod
            def from_pretrained(cls, mid, **kw):
                cls.calls.append(kw)
                # first: CPU load returns meta; second (device_map) raises; third succeeds
                if len(cls.calls) == 1:
                    return _P()
                if len(cls.calls) == 2:
                    raise NotImplementedError("meta still on device_map path")
                return _P3()

        diffusers = types.SimpleNamespace(DiffusionPipeline=_DP)
        sys.modules['diffusers'] = diffusers

        import flux_local as fl
        fl.CURRENT_MODEL_ID = None
        fl._get_model_id = lambda: 'dummy/model2'  # type: ignore
        fl._free_pipe()
        pipe = fl._ensure_pipe(None)
        self.assertIsInstance(pipe, _P3)
        # Third call should be low_cpu_mem_usage=True and no device_map
        self.assertEqual(len(_DP.calls), 3)
        self.assertTrue(_DP.calls[2].get('low_cpu_mem_usage') is True)
        self.assertIsNone(_DP.calls[2].get('device_map'))

    def test_device_map_value_error_still_falls_back(self):
        """ValueError from device_map path should trigger the CPU reload fallback."""
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            float16='fp16',
        )
        sys.modules['torch'] = torch

        class _P:
            def __init__(self):
                self.safety_checker = None
                self.feature_extractor = None
                self.scheduler = types.SimpleNamespace(init_noise_sigma=1.0)
            def to(self, *a, **k):
                raise NotImplementedError("meta first")

        class _P3(_P):
            def to(self, *a, **k):
                return self

        class _DP:
            calls = []
            @classmethod
            def from_pretrained(cls, mid, **kw):
                cls.calls.append(kw)
                if len(cls.calls) == 1:
                    return _P()
                if len(cls.calls) == 2:
                    raise ValueError("device_map must be a string")
                return _P3()

        diffusers = types.SimpleNamespace(DiffusionPipeline=_DP)
        sys.modules['diffusers'] = diffusers

        import flux_local as fl
        fl.CURRENT_MODEL_ID = None
        fl._get_model_id = lambda: 'dummy/model3'  # type: ignore
        fl._free_pipe()
        pipe = fl._ensure_pipe(None)
        self.assertIsInstance(pipe, _P3)
        self.assertEqual(len(_DP.calls), 3)
        self.assertTrue(_DP.calls[2].get('low_cpu_mem_usage') is True)
        self.assertIsNone(_DP.calls[2].get('device_map'))


if __name__ == '__main__':
    unittest.main()
