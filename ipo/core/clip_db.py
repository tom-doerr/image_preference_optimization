"""PostgreSQL persistence for CLIP preference samples."""
import os
import numpy as np
from datetime import datetime

DB_URL = os.environ.get("CLIP_DB_URL", "postgresql://tom@/clip_preferences")
_ENGINE = None


def get_engine():
    """Get SQLAlchemy engine (cached)."""
    global _ENGINE
    if _ENGINE is None:
        from sqlalchemy import create_engine
        _ENGINE = create_engine(DB_URL)
    return _ENGINE


def init_db():
    """Create samples table if not exists."""
    from sqlalchemy import text
    sql = """CREATE TABLE IF NOT EXISTS samples (
        id SERIAL PRIMARY KEY, prompt TEXT, embedding BYTEA,
        label INTEGER, image_data BYTEA, created_at TIMESTAMP DEFAULT NOW())"""
    with get_engine().connect() as c:
        c.execute(text(sql))
        c.commit()


def save_sample(prompt: str, emb: np.ndarray, label: int, img_data: bytes):
    """Save a labeled sample to DB."""
    from sqlalchemy import text
    with get_engine().connect() as c:
        c.execute(text(
            "INSERT INTO samples (prompt, embedding, label, image_data) VALUES (:p, :e, :l, :i)"
        ), {"p": prompt, "e": emb.tobytes(), "l": label, "i": img_data})
        c.commit()


def get_samples(prompt: str = None):
    """Get samples as (X, y) arrays."""
    from sqlalchemy import text
    sql = "SELECT embedding, label FROM samples"
    if prompt:
        sql += " WHERE prompt = :p"
    with get_engine().connect() as c:
        rows = c.execute(text(sql), {"p": prompt} if prompt else {}).fetchall()
    if not rows:
        return np.zeros((0, 768)), np.zeros((0,))
    X = np.array([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    return X, np.array([r[1] for r in rows])
