import io
import hashlib
from datetime import datetime, timezone
from typing import Any
import os
import shutil
import numpy as np
from constants import APP_VERSION
from latent_opt import dumps_state


def state_path_for_prompt(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return f"latent_state_{h}.npz"

def data_root_for_prompt(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return os.path.join("data", h)


def dataset_rows_for_prompt(prompt: str) -> int:
    """Count rows from per-sample folders only (data/<hash>/*/sample.npz)."""
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    root = os.path.join("data", h)
    if not os.path.isdir(root):
        return 0
    n = 0
    for name in os.listdir(root):
        sample_path = os.path.join(root, name, "sample.npz")
        if os.path.exists(sample_path):
            n += 1
    return n

def dataset_rows_for_prompt_dim(prompt: str, d: int) -> int:
    """Count rows in folders where X.shape[1] == d."""
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    root = os.path.join("data", h)
    if not os.path.isdir(root):
        return 0
    n = 0
    for name in os.listdir(root):
        sample_path = os.path.join(root, name, "sample.npz")
        if not os.path.exists(sample_path):
            continue
        with np.load(sample_path) as z:
            X = z['X'] if 'X' in z.files else None
            if X is not None and getattr(X, 'ndim', 1) == 2 and X.shape[1] == int(d):
                n += 1
    return n

def dataset_rows_all_for_prompt(prompt: str) -> int:
    """Rows across all dims from folders (same as dataset_rows_for_prompt)."""
    return dataset_rows_for_prompt(prompt)


def get_dataset_for_prompt_or_session(prompt: str, session_state) -> tuple[object | None, object | None]:
    """Return (X, y) from per-sample folders only (no NPZ fallback).

    Reads rows from data/<hash>/*/sample.npz and concatenates them. If the
    folder is missing or empty, returns (None, None).
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
    # Write only to per-sample folder dataset
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    root = os.path.join("data", h)
    os.makedirs(root, exist_ok=True)
    # Determine next row index from existing sample folders, then atomically
    # create a new directory to avoid races when many saves happen quickly.
    try:
        import re as _re
        idxs = []
        for name in os.listdir(root):
            m = _re.fullmatch(r"(\d+)", name)
            if m:
                idxs.append(int(m.group(1)))
        next_idx = (max(idxs) + 1) if idxs else 1
    except Exception:
        next_idx = 1
    # Attempt to create a unique directory by incrementing until it succeeds.
    # Keeps names numeric for simplicity and test compatibility.
    attempts = 0
    while True:
        attempts += 1
        sample_dir = os.path.join(root, f"{next_idx:06d}")
        try:
            os.mkdir(sample_dir)
            break
        except FileExistsError:
            next_idx += 1
            if attempts > 1000:
                # Failsafe: give up with a new high index if something is odd
                next_idx += 1000
        except Exception:
            # Best effort: try the next index
            next_idx += 1
    sample_path = os.path.join(sample_dir, "sample.npz")
    np.savez_compressed(sample_path, X=feat, y=np.array([label], dtype=float))
    # Maintain a minimal backup of the sample file (legacy backup test)
    try:
    base = f"sample_{h}_{next_idx:06d}.npz"
        tmp_copy = os.path.join('.', base)
        # create a copy named deterministically for backup routine
        import shutil as _sh
        _sh.copy2(sample_path, tmp_copy)
        _write_backups(tmp_copy)
        try:
            os.remove(tmp_copy)
        except Exception:
            pass
    except Exception:
        pass

    # Also save per-sample NPZ under data/<hash>/<row_idx>/sample.npz
    return next_idx


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
    """Folder-based stats: rows/pos/neg/d/recent_labels from data/<hash>/*/sample.npz."""
    rows = pos = neg = d = 0
    recent: list[int] = []
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    root = os.path.join("data", h)
    if not os.path.isdir(root):
        return {"rows": 0, "pos": 0, "neg": 0, "d": 0, "recent_labels": recent}
    labels: list[int] = []
    for name in sorted(os.listdir(root)):
        sample_path = os.path.join(root, name, "sample.npz")
        if not os.path.exists(sample_path):
            continue
        with np.load(sample_path) as z:
            Xi = z['X'] if 'X' in z.files else None
            yi = z['y'] if 'y' in z.files else None
        if Xi is None or yi is None:
            continue
        if d == 0 and getattr(Xi, 'ndim', 1) == 2:
            d = int(Xi.shape[1])
        rows += 1
        val = int(np.asarray(yi).astype(int).ravel()[0])
        labels.append(val)
    if labels:
        arr = np.asarray(labels)
        pos = int((arr > 0).sum())
        neg = int((arr < 0).sum())
        recent = [int(v) for v in arr[-5:]]
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
