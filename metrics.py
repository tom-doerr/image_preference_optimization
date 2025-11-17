import numpy as np
from typing import Dict


def pair_metrics(w: np.ndarray, z_a: np.ndarray, z_b: np.ndarray) -> Dict[str, float | str]:
    za_n = float(np.linalg.norm(z_a))
    zb_n = float(np.linalg.norm(z_b))
    diff = z_b - z_a
    diff_n = float(np.linalg.norm(diff))
    w_n = float(np.linalg.norm(w))
    if w_n > 0.0 and diff_n > 0.0:
        c = float(np.dot(w, diff) / (w_n * diff_n))
    else:
        c = float("nan")
    return {
        "za_norm": za_n,
        "zb_norm": zb_n,
        "diff_norm": diff_n,
        "cos_w_diff": c,
    }

