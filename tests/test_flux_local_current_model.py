import os
import sys
import types
import unittest


class TestFluxLocalPrefersCurrentModel(unittest.TestCase):
    def test_ensure_pipe_uses_current_model_when_env_missing(self):
        # Remove env var if present
        if "FLUX_LOCAL_MODEL" in os.environ:
            del os.environ["FLUX_LOCAL_MODEL"]

        # Create import-time stubs for torch and diffusers before calling _ensure_pipe
        class _Cuda:
            @staticmethod
            def is_available():
                return True

        torch_stub = types.SimpleNamespace(cuda=_Cuda(), float16=object())

        class DummyPipe:
            def to(self, device):
                return self

        class DummyDP:
            @staticmethod
            def from_pretrained(mid, **kw):
                return DummyPipe()

        diffusers_stub = types.SimpleNamespace(DiffusionPipeline=DummyDP)
        sys.modules['torch'] = torch_stub  # type: ignore[assignment]
        sys.modules['diffusers'] = diffusers_stub  # type: ignore[assignment]

        import flux_local as fl

        # Set CURRENT_MODEL_ID and ensure call does not read env or raise
        fl.CURRENT_MODEL_ID = "stabilityai/sd-turbo"
        fl.PIPE = None
        try:
            fl._ensure_pipe(None)
        except Exception as e:  # pragma: no cover - test must not raise
            self.fail(f"_ensure_pipe raised unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
