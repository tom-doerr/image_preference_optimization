import io
import hashlib
from datetime import datetime, timezone
from typing import Any
import numpy as np
from constants import APP_VERSION
from latent_opt import dumps_state


def state_path_for_prompt(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return f"latent_state_{h}.npz"


def dataset_path_for_prompt(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return f"dataset_{h}.npz"


def dataset_rows_for_prompt(prompt: str) -> int:
    """Return number of rows in the saved dataset for this prompt, or 0 if none."""
    import os
    p = dataset_path_for_prompt(prompt)
    if not os.path.exists(p):
        return 0
    with np.load(p) as d:
        if 'X' not in d.files:
            return 0
        X = d['X']
        return int(getattr(X, 'shape', (0,))[0])


def append_dataset_row(prompt: str, feat: np.ndarray, label: float) -> int:
    """Append one (feat, label) to the dataset NPZ for this prompt.

    Returns the new number of rows.
    """
    import os
    p = dataset_path_for_prompt(prompt)
    if os.path.exists(p):
        with np.load(p) as d:
            Xd = d['X'] if 'X' in d.files else np.zeros((0, feat.shape[1]))
            yd = d['y'] if 'y' in d.files else np.zeros((0,))
    else:
        Xd = np.zeros((0, feat.shape[1]))
        yd = np.zeros((0,))
    X_new = np.vstack([Xd, feat]) if Xd.size else feat
    y_new = np.concatenate([yd, np.array([label], dtype=float)]) if yd.size else np.array([label], dtype=float)
    np.savez_compressed(p, X=X_new, y=y_new)
    return int(X_new.shape[0])


def export_state_bytes(state, prompt: str) -> bytes:
    raw = dumps_state(state)
    with np.load(io.BytesIO(raw)) as data:
        items = {k: data[k] for k in data.files}
    items["prompt"] = np.array(prompt)
    items["created_at"] = np.array(datetime.now(timezone.utc).isoformat())
    items["app_version"] = np.array(APP_VERSION)
    buf = io.BytesIO()
    np.savez_compressed(buf, **items)
    return buf.getvalue()


def read_metadata(path: str) -> dict:
    """Return minimal metadata dict from a saved NPZ state file.

    Keys: 'app_version' (str|None), 'created_at' (str|None), 'prompt' (str|None)
    Missing keys return as None. Errors bubble up to caller (keep it simple).
    """
    out = {"app_version": None, "created_at": None, "prompt": None}
    with np.load(path) as data:
        if 'app_version' in data.files:
            out['app_version'] = data['app_version'].item()
        if 'created_at' in data.files:
            out['created_at'] = data['created_at'].item()
        if 'prompt' in data.files:
            try:
                out['prompt'] = data['prompt'].item()
            except Exception:
                pass
    return out
