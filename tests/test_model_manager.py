"""Tests for ModelManager."""
import pytest


def test_model_manager_imports():
    """ModelManager can be imported."""
    from ipo.core.model_manager import ModelManager, LocalBackend, ServerBackend
    assert ModelManager is not None


def test_model_id_resolution():
    """Model IDs are resolved correctly."""
    from ipo.infra.pipeline_local import _resolve_model_id
    assert _resolve_model_id("sd-turbo") == "stabilityai/sd-turbo"
    assert _resolve_model_id("flux-schnell") == "black-forest-labs/FLUX.1-schnell"


def test_gen_client_caching():
    """GenerationClient caches model to avoid redundant calls."""
    from ipo.server.gen_client import GenerationClient
    c = GenerationClient("http://fake:8580")
    c._current_model = "sd-turbo"
    assert c.set_model("sd-turbo")["status"] == "cached"
