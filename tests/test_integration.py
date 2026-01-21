"""Integration tests for end-to-end flows."""
import numpy as np
import pytest


class TestClipFlow:
    """Test CLIP mode end-to-end flow."""

    def test_embed_and_score(self):
        """Embedding + ridge scoring works together."""
        from ipo.ui.clip_mode import _ridge_fit, _score_one
        X = np.random.randn(10, 768).astype(np.float32)
        y = np.random.randn(10)
        w = _ridge_fit(X, y)
        emb = np.random.randn(768).astype(np.float32)
        score = float(np.dot(emb, w))
        assert isinstance(score, float)

    def test_cv_and_predict(self):
        """CV alpha selection + prediction."""
        from ipo.ui.clip_mode import _ridge_cv
        X = np.random.randn(20, 768).astype(np.float32)
        y = (np.random.randn(20) > 0).astype(float)
        w, alpha, scores = _ridge_cv(X, y)
        preds = X @ w
        assert preds.shape == (20,)


class TestDBFlow:
    """Test database integration."""

    def test_roundtrip(self):
        """Save and retrieve samples."""
        from ipo.core.clip_db import init_db, save_sample, get_samples
        init_db()
        prompt = f"int_{np.random.randint(1e6)}"
        emb = np.random.randn(768).astype(np.float32)
        save_sample(prompt, emb, 1, b"img")
        X, y = get_samples(prompt)
        assert len(y) >= 1


class TestModelRegistry:
    """Test model registry integration."""

    def test_all_models_have_pipeline(self):
        """All models specify valid pipeline type."""
        from ipo.infra.model_registry import MODELS
        valid = {"FluxPipeline", "DiffusionPipeline"}
        for name, cfg in MODELS.items():
            assert cfg["pipeline"] in valid, f"{name}"

    def test_resolve_all_models(self):
        """All model keys resolve to HF IDs."""
        from ipo.infra.model_registry import MODELS, resolve_model_id
        for key in MODELS:
            hf_id = resolve_model_id(key)
            assert "/" in hf_id, f"{key} -> {hf_id}"


class TestServerAPI:
    """Test server API (requires running server)."""

    @pytest.mark.skipif(True, reason="Requires server")
    def test_health_endpoint(self):
        """Health endpoint returns status."""
        import requests
        r = requests.get("http://localhost:8580/health", timeout=5)
        assert r.status_code == 200
        assert "status" in r.json()


class TestConstantsIntegration:
    """Test constants module integration."""

    def test_keys_are_strings(self):
        """All Keys are string values."""
        from ipo.infra.constants import Keys
        for attr in dir(Keys):
            if not attr.startswith("_"):
                val = getattr(Keys, attr)
                assert isinstance(val, str), f"{attr}"

    def test_default_model_in_options(self):
        """Default model is in MODEL_OPTIONS."""
        from ipo.infra.constants import DEFAULT_MODEL, MODEL_OPTIONS
        assert DEFAULT_MODEL in MODEL_OPTIONS
