"""Integration tests for CLIP DB."""
import numpy as np
import pytest
from PIL import Image

from ipo.core.clip_db import init_db, save_sample, get_samples


def _make_img():
    """Create test image."""
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))


def _img_bytes(img):
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_init_db():
    """Test DB initialization."""
    init_db()  # Should not raise


def test_save_and_get():
    """Test save and retrieve samples."""
    init_db()
    emb = np.random.randn(768).astype(np.float32)
    img = _make_img()
    save_sample("test_prompt", emb, 1, _img_bytes(img))
    X, y = get_samples("test_prompt")
    assert X.shape[1] == 768
    assert len(y) >= 1
