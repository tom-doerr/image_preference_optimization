"""Tests for generation server/client."""
import pytest


def test_gen_client_imports():
    """GenerationClient can be imported."""
    from ipo.server.gen_client import GenerationClient
    assert GenerationClient is not None


def test_gen_client_init():
    """GenerationClient initializes with URL."""
    from ipo.server.gen_client import GenerationClient
    c = GenerationClient("http://test:8580")
    assert c.base_url == "http://test:8580"


def test_gen_client_cache_hit():
    """GenerationClient returns cached for same model."""
    from ipo.server.gen_client import GenerationClient
    c = GenerationClient("http://fake:8580")
    c._current_model = "test-model"
    result = c.set_model("test-model")
    assert result["status"] == "cached"
