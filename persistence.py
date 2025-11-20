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
    """Return (X, y) for this prompt from disk-backed data.

    Prefer per-sample folders under data/<hash>/ when present; otherwise fall
    back to the aggregate dataset_<hash>.npz file. session_state is unused.
    """
    X = y = None
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    root = os.path.join("data", h)
    try:
        if os.path.isdir(root):
            xs = []
            ys = []
            for name in sorted(os.listdir(root)):
                sample_path = os.path.join(root, name, "sample.npz")
                if not os.path.exists(sample_path):
                    continue
                with np.load(sample_path) as d:
                    Xi = d["X"] if "X" in d.files else None
                    yi = d["y"] if "y" in d.files else None
                if Xi is None or yi is None:
                    continue
                xs.append(Xi)
                ys.append(yi)
            if xs:
                X = np.vstack(xs)
                y = np.concatenate(ys)
                try:
                    print(f"[data] loaded {X.shape[0]} rows d={X.shape[1]} from data/{h}")
                except Exception:
                    pass
                return X, y
    except Exception:
        X = y = None
    # Fallback: aggregate NPZ
    p = dataset_path_for_prompt(prompt)
    try:
        if os.path.exists(p):
            with np.load(p) as d:
                X = d["X"] if "X" in d.files else None
                y = d["y"] if "y" in d.files else None
            if X is not None and y is not None:
                try:
                    print(f"[data] loaded {X.shape[0]} rows d={X.shape[1]} from {os.path.basename(p)}")
                except Exception:
                    pass
    except Exception:
        X = y = None
    if X is None or y is None:
        try:
            print(f"[data] no dataset for prompt={prompt!r}")
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
        # If feature dimension changed (e.g., resolution changed), start a fresh
        # aggregate file for the new dimension instead of failing a vstack.
        try:
            if Xd.size and getattr(Xd, 'shape', (0, 0))[1] != int(feat.shape[1]):
                Xd = np.zeros((0, feat.shape[1]))
                yd = np.zeros((0,))
        except Exception:
            # Keep minimal; on shape introspection error just fall back to fresh.
            Xd = np.zeros((0, feat.shape[1]))
            yd = np.zeros((0,))
    else:
        Xd = np.zeros((0, feat.shape[1]))
        yd = np.zeros((0,))
    X_new = np.vstack([Xd, feat]) if Xd.size else feat
    y_new = np.concatenate([yd, np.array([label], dtype=float)]) if yd.size else np.array([label], dtype=float)
    np.savez_compressed(p, X=X_new, y=y_new)
    _write_backups(p)

    # Also save per-sample NPZ under data/<hash>/<row_idx>/sample.npz
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    root = os.path.join("data", h)
    row_idx = int(X_new.shape[0])
    sample_dir = os.path.join(root, f"{row_idx:06d}")
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, "sample.npz")
    np.savez_compressed(sample_path, X=feat, y=np.array([label], dtype=float))

    return row_idx


def save_sample_image(prompt: str, row_idx: int, img: Any) -> None:
    """Save a sample image alongside its feature/label NPZ in the data folder.

    Keeps the image in `data/<hash>/<row_idx>/image.png`. Errors are not raised.
    """
    try:
        h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        root = os.path.join("data", h)
        sample_dir = os.path.join(root, f"{row_idx:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        path = os.path.join(sample_dir, "image.png")
        import numpy as _np
        from PIL import Image  # pillow is already a dependency via diffusers/Streamlit
        if hasattr(img, "save"):
            img.save(path)
        else:
            arr = _np.asarray(img)
            Image.fromarray(arr).save(path)
    except Exception:
        # Image saving is best-effort; do not affect core flow.
        pass


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
