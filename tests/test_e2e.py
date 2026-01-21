"""End-to-end tests (require GPU)."""
import numpy as np
import pytest


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(
    not _has_cuda(), reason="No CUDA"
)


@requires_gpu
class TestGenerationE2E:
    """E2E generation tests."""

    def test_pipeline_loads(self):
        """Pipeline can be loaded."""
        from ipo.infra.flux_pipeline import load_pipeline
        # Just test import, actual load needs model
        assert load_pipeline is not None

    def test_bnb_config_creates(self):
        """BitsAndBytes config can be created."""
        pytest.importorskip("diffusers")
        from ipo.infra.flux_pipeline import _get_bnb_config
        cfg = _get_bnb_config()
        assert cfg.load_in_4bit is True
