"""Tests for flux_pipeline module."""
import os
import pytest


def test_get_token_from_hf_token():
    """_get_token returns HF_TOKEN env var."""
    from ipo.infra.flux_pipeline import _get_token
    old = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = "test_token_123"
    try:
        assert _get_token() == "test_token_123"
    finally:
        if old:
            os.environ["HF_TOKEN"] = old
        else:
            os.environ.pop("HF_TOKEN", None)


def test_model_registry_keys():
    """Model registry entries have required keys."""
    from ipo.infra.model_registry import MODELS
    for name, cfg in MODELS.items():
        assert "hf_id" in cfg, f"{name} missing hf_id"
        assert "pipeline" in cfg, f"{name} missing pipeline"


def test_resolve_model_id():
    """resolve_model_id maps short names to HF IDs."""
    from ipo.infra.model_registry import resolve_model_id
    assert resolve_model_id("sd-turbo") == "stabilityai/sd-turbo"
    assert resolve_model_id("flux-schnell") == "black-forest-labs/FLUX.1-schnell"
    assert resolve_model_id("unknown/model") == "unknown/model"


def test_model_options_list():
    """MODEL_OPTIONS is a list of model keys."""
    from ipo.infra.model_registry import MODEL_OPTIONS
    assert isinstance(MODEL_OPTIONS, list)
    assert "sd-turbo" in MODEL_OPTIONS
