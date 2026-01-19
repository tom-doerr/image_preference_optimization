"""Data curation and labeling functions."""
import logging
import numpy as np
from typing import Optional, Any

log = logging.getLogger(__name__)


def add_sample(
    prompt: str, z: np.ndarray, z_base: np.ndarray, label: int, img: Optional[Any] = None
) -> None:
    """Add a labeled sample to the dataset."""
    from ipo.core.persistence import append_sample
    feat = (z - z_base).reshape(1, -1)
    append_sample(prompt, feat, float(label), img)
