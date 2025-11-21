import sys
import types
import unittest


class TestSafetyCheckerDisabled(unittest.TestCase):
    def tearDown(self):
        for m in ("flux_local", "diffusers", "torch"):
            sys.modules.pop(m, None)

    def test_set_model_disables_safety_checker(self):
        # Stub torch with CUDA available
        class _Cuda:
            @staticmethod
            def is_available():
                return True

        torch = types.SimpleNamespace(cuda=_Cuda(), float16=None)
        sys.modules["torch"] = torch

        # Minimal DiffusionPipeline stub
        class _Cfg:
            def __init__(self):
                self.requires_safety_checker = True

        class _Pipe:
            def __init__(self):
                self.safety_checker = object()
                self.feature_extractor = object()
                self.config = _Cfg()
                self.scheduler = types.SimpleNamespace(config={})

            def to(self, *a, **k):
                return self

            def enable_attention_slicing(self):
                return None

            def enable_vae_slicing(self):
                return None

            def enable_xformers_memory_efficient_attention(self):
                return None

            def register_to_config(self, **cfg):
                if "requires_safety_checker" in cfg:
                    self.config.requires_safety_checker = bool(
                        cfg["requires_safety_checker"]
                    )

        class _DP:
            @staticmethod
            def from_pretrained(*a, **k):
                return _Pipe()

        diffusers = types.SimpleNamespace(DiffusionPipeline=_DP)
        sys.modules["diffusers"] = diffusers

        import flux_local

        flux_local.set_model("dummy/model")
        pipe = flux_local.PIPE
        self.assertIsNotNone(pipe)
        self.assertIsNone(pipe.safety_checker)
        self.assertFalse(getattr(pipe.config, "requires_safety_checker", True))


if __name__ == "__main__":
    unittest.main()

