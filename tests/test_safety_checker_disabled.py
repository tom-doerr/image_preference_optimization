import sys
import types
import unittest


class PipeStub:
    def __init__(self):
        self.safety_checker = object()
        self.feature_extractor = object()
        self.requires_safety_checker = True
        self.config = types.SimpleNamespace(requires_safety_checker=True)

    def to(self, *_a, **_k):
        return self

    def register_to_config(self, **kw):
        if "requires_safety_checker" in kw:
            self.requires_safety_checker = bool(kw["requires_safety_checker"])  # type: ignore


class TestSafetyCheckerDisabled(unittest.TestCase):
    def test_set_model_disables_filter(self):
        # Stub torch + diffusers to avoid real deps
        torch = types.ModuleType("torch")
        torch.float16 = object()
        torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        sys.modules["torch"] = torch

        diff = types.ModuleType("diffusers")

        class DiffusionPipeline:
            @classmethod
            def from_pretrained(cls, *a, **k):
                return PipeStub()

        diff.DiffusionPipeline = DiffusionPipeline
        sys.modules["diffusers"] = diff

        import flux_local as fl

        # Use a dummy model id and load once
        fl.set_model("dummy/model")
        p = fl.PIPE
        self.assertIsNone(p.safety_checker)
        self.assertIsNone(p.feature_extractor)
        # Both attribute and config flag should show disabled
        self.assertFalse(getattr(p, "requires_safety_checker", False))
        self.assertFalse(getattr(p.config, "requires_safety_checker", True))


if __name__ == "__main__":
    unittest.main()

