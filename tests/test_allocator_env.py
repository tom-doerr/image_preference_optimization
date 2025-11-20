import os
import sys
import types
import unittest
import importlib


class TestAllocatorEnv(unittest.TestCase):
    def setUp(self):
        for m in ("torch", "diffusers", "flux_local"):
            if m in sys.modules:
                del sys.modules[m]

    def test_allocator_env_set_on_load(self):
        # Mock torch/diffusers
        torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available():
                return True

        torch.cuda = _Cuda()
        torch.float16 = object()
        sys.modules["torch"] = torch

        calls = {"n": 0}
        diffusers = types.ModuleType("diffusers")

        class _Pipe:
            def to(self, device):
                return self

        class _DP:
            @staticmethod
            def from_pretrained(mid, **kwargs):
                calls["n"] += 1
                return _Pipe()

        diffusers.DiffusionPipeline = _DP
        sys.modules["diffusers"] = diffusers

        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["FLUX_LOCAL_MODEL"] = "dummy/model"
        import flux_local

        importlib.reload(flux_local)
        flux_local.set_model("dummy/model")
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", os.environ)
        self.assertTrue(
            os.environ["PYTORCH_CUDA_ALLOC_CONF"].startswith("expandable_segments")
        )


if __name__ == "__main__":
    unittest.main()
