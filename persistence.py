import io
import hashlib
from datetime import datetime, timezone
from typing import Any
import os
import shutil
from datetime import datetime, timezone
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


def get_dataset_for_prompt_or_session(prompt: str, session_state) -> tuple[object | None, object | None]:
    """Return (X, y) from the saved dataset for this prompt, or from session_state.

    Prefers the persisted NPZ; if missing, uses `session_state.dataset_X`/`dataset_y` when present.
    Keeps code minimal and avoids repeating the same load-or-session pattern.
    """
    X = y = None
    p = dataset_path_for_prompt(prompt)
    try:
        if os.path.exists(p):
            with np.load(p) as d:
                X = d['X'] if 'X' in d.files else None
                y = d['y'] if 'y' in d.files else None
    except Exception:
        X = y = None
    if (X is None or y is None) and getattr(session_state, 'dataset_X', None) is not None:
        try:
            X = session_state.dataset_X
            y = session_state.dataset_y
        except Exception:
            pass
    return X, y


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
    _write_backups(p)
    return int(X_new.shape[0])


def _write_backups(path: str) -> None:
    """Write simple time-bucketed backups for the dataset file.

    Creates/overwrites three snapshots per call:
    - backups/minutely/<base>.<YYYYMMDD_HHMM>.npz
    - backups/hourly/<base>.<YYYYMMDD_HH>.npz
    - backups/daily/<base>.<YYYYMMDD>.npz
    Minimal and synchronous by design.
    """
    try:
        now = datetime.now(timezone.utc)
        base = os.path.basename(path)
        root = os.path.dirname(path) or "."
        buckets = {
            os.path.join(root, "backups", "minutely"): now.strftime("%Y%m%d_%H%M"),
            os.path.join(root, "backups", "hourly"): now.strftime("%Y%m%d_%H"),
            os.path.join(root, "backups", "daily"): now.strftime("%Y%m%d"),
        }
        for folder, stamp in buckets.items():
            os.makedirs(folder, exist_ok=True)
            dst = os.path.join(folder, f"{base}.{stamp}.npz")
            shutil.copy2(path, dst)
    except Exception:
        # Minimal: don't hide errors on save, but backups are best-effort.
        pass


def dataset_stats_for_prompt(prompt: str) -> dict:
    """Return minimal stats for the saved dataset of this prompt.

    Keys: rows, pos, neg, d, recent_labels (list of ints, up to 5)
    """
    rows = pos = neg = d = 0
    recent = []
    import os
    p = dataset_path_for_prompt(prompt)
    if not os.path.exists(p):
        return {"rows": 0, "pos": 0, "neg": 0, "d": 0, "recent_labels": recent}
    with np.load(p) as z:
        X = z['X'] if 'X' in z.files else None
        y = z['y'] if 'y' in z.files else None
        if X is not None and hasattr(X, 'shape'):
            rows = int(X.shape[0])
            d = int(X.shape[1]) if X.ndim == 2 else 0
        if y is not None:
            yy = np.asarray(y).astype(int)
            pos = int((yy > 0).sum())
            neg = int((yy < 0).sum())
            recent = [int(v) for v in yy[-5:]]
    return {"rows": rows, "pos": pos, "neg": neg, "d": d, "recent_labels": recent}


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
