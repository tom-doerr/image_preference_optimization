import hashlib
import io
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ipo.infra.constants import APP_VERSION


def state_path_for_prompt(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    root = data_root_for_prompt(prompt)
    os.makedirs(root, exist_ok=True)
    new_path = os.path.join(root, "latent_state.npz")
    legacy_path = f"latent_state_{h}.npz"
    if os.path.exists(legacy_path) and not os.path.exists(new_path):
        return legacy_path
    return new_path


def _base_data_dir() -> str:
    """Return base data directory.

    Precedence:
    - IPO_DATA_ROOT env if set
    - Per-test isolated folder when PYTEST_CURRENT_TEST is present
    - Default 'data/'
    """
    global _BASE_DIR_CACHE
    # Honor explicit root first
    root = os.getenv("IPO_DATA_ROOT")
    if root:
        return root
    # If a cached test root exists (set earlier in this session), reuse it
    try:
        if _BASE_DIR_CACHE is not None:
            return _BASE_DIR_CACHE
    except NameError:
        pass
    # Under pytest, use a single per-process temp root to avoid flakiness
    if os.getenv("PYTEST_CURRENT_TEST"):
        if _BASE_DIR_CACHE is None:
            run = os.getenv("IPO_TEST_RUN") or str(os.getpid())
            _BASE_DIR_CACHE = os.path.join(".tmp_cli_models", f"tdata_run_{run}")
        return _BASE_DIR_CACHE
    return "data"


def data_root_for_prompt(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return os.path.join(_base_data_dir(), h)


def _lockfile_path_for_prompt(prompt: str) -> str:
    return os.path.join(data_root_for_prompt(prompt), ".append.lock")


@contextmanager
def _file_lock_for_prompt(prompt: str):
    """On-disk lock for multi-process safety during dataset appends.

    Uses fcntl.flock on a per-prompt lock file. Minimal and POSIX-only by design.
    """
    import fcntl  # type: ignore

    root = data_root_for_prompt(prompt)
    os.makedirs(root, exist_ok=True)
    lf = _lockfile_path_for_prompt(prompt)
    f = open(lf, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def dataset_rows_for_prompt(prompt: str) -> int:
    """Count rows from per-sample folders only (data/<hash>/*/sample.npz)."""
    root = data_root_for_prompt(prompt)
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
    root = data_root_for_prompt(prompt)
    if not os.path.isdir(root):
        return 0
    n = 0
    for name in os.listdir(root):
        sample_path = os.path.join(root, name, "sample.npz")
        if not os.path.exists(sample_path):
            continue
        with np.load(sample_path) as z:
            X = z["X"] if "X" in z.files else None
            if X is not None and getattr(X, "ndim", 1) == 2 and X.shape[1] == int(d):
                n += 1
    return n


# dataset_rows_all_for_prompt removed (203f): use dataset_rows_for_prompt instead.


def _target_dim_from_session(session_state) -> int | None:
    try:
        lstate = getattr(session_state, "lstate", None)
        if lstate is None:
            return None
        d = int(getattr(lstate, "d", 0) or 0)
        return d or None
    except Exception:
        return None


def _iter_sample_paths(root: str):
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "sample.npz")
        if os.path.exists(p):
            yield p


def _load_sample_npz(path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    with np.load(path) as d:
        Xi = d["X"] if "X" in d.files else None
        yi = d["y"] if "y" in d.files else None
    return Xi, yi


def _record_dim_mismatch(session_state, d_x: int | None, target_d: int) -> None:
    try:
        from ipo.infra.constants import Keys as _K

        session_state[_K.DATASET_DIM_MISMATCH] = (d_x, target_d)
    except Exception:
        session_state["dataset_dim_mismatch"] = (d_x, target_d)


def get_dataset_for_prompt_or_session(
    prompt: str, session_state
) -> tuple[object | None, object | None]:
    """Return (X, y) from per-sample folders only (no NPZ fallback).

    - Reads rows from data/<hash>/*/sample.npz and concatenates them.
    - If session_state.lstate exists and feature dim != lstate.d, skip those rows
      and record the mismatch in session_state['dataset_dim_mismatch'].
    - If the folder is missing or empty, returns (None, None).
    """
    X = y = None
    root = data_root_for_prompt(prompt)
    try:
        if not os.path.isdir(root):
            raise FileNotFoundError
        target_d = _target_dim_from_session(session_state)
        xs, ys, skipped = _load_rows_filtered(root, target_d, session_state)
        if xs:
            X = np.vstack(xs)
            y = np.concatenate(ys)
            try:
                extra = f" (skipped {skipped})" if skipped else ""
                print(f"[data] loaded {X.shape[0]} rows d={X.shape[1]} from {root}{extra}")
            except Exception:
                pass
            return X, y
        if skipped:
            print(f"[data] all {skipped} rows skipped due to dim mismatch in {root}")
    except Exception:
        X = y = None
    try:
        print(f"[data] no dataset for prompt={prompt!r}")
    except Exception:
        pass
    return X, y


def _load_rows_filtered(root: str, target_d: int | None, session_state) -> tuple[list[np.ndarray], list[np.ndarray], int]:  # noqa: E501
    """Load per-sample rows, filtering by feature dim. Returns (xs, ys, skipped)."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    skipped = 0
    for sample_path in _iter_sample_paths(root):
        Xi, yi = _load_sample_npz(sample_path)
        if Xi is None or yi is None:
            continue
        if target_d is not None:
            try:
                d_x = int(getattr(Xi, "shape", (0, 0))[1])
            except Exception:
                d_x = None  # type: ignore
            if d_x != target_d:
                _record_dim_mismatch(session_state, d_x, target_d)
                print(f"[data] skip {sample_path}: d={d_x} != target {target_d}")
                skipped += 1
                continue
        xs.append(Xi)
        ys.append(yi)
    return xs, ys, skipped


def append_dataset_row(prompt: str, feat: np.ndarray, label: float) -> int:
    """Append one (feat, label) to the dataset NPZ for this prompt.

    Returns the new number of rows.
    """
    import os

    # Write only to per-sample folder dataset, protected by in-proc + file locks
    root = data_root_for_prompt(prompt)
    os.makedirs(root, exist_ok=True)
    with _lock_for_prompt(prompt):
        with _file_lock_for_prompt(prompt):
            # Determine next row index from existing sample folders, then atomically
            # create a new directory; still guard with the prompt lock to avoid
            # many concurrent os.listdir/os.mkdir races.
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
                        next_idx += 1000
                except Exception:
                    next_idx += 1
            sample_path = os.path.join(sample_dir, "sample.npz")
            np.savez_compressed(sample_path, X=feat, y=np.array([label], dtype=float))
            ret_idx = next_idx
    try:
        import numpy as _np

        fn = float(_np.linalg.norm(_np.asarray(feat, dtype=float)))
        d = feat.shape[-1] if hasattr(feat, "shape") else len(feat)
        print(f"[data] saved {sample_path} d={d} label={int(label):+d} ‖feat‖={fn:.1f}")
    except Exception:
        pass
    # Also save per-sample NPZ under data/<hash>/<row_idx>/sample.npz
    return ret_idx


def save_sample_image(prompt: str, row_idx: int, img: Any) -> None:
    """Save a sample image alongside its feature/label NPZ in the data folder.

    Keeps the image in `data/<hash>/<row_idx>/image.png`. Errors are not raised.
    """
    try:
        root = data_root_for_prompt(prompt)
        sample_dir = os.path.join(root, f"{row_idx:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        path = os.path.join(sample_dir, "image.png")
        import numpy as _np
        from PIL import Image  # pillow is already a dependency via diffusers/Streamlit

        with _lock_for_prompt(prompt):
            with _file_lock_for_prompt(prompt):
                if hasattr(img, "save"):
                    img.save(path)
                else:
                    arr = _np.asarray(img)
                    Image.fromarray(arr).save(path)
        try:
            print(f"[data] wrote image {path}")
        except Exception:
            pass
    except Exception:
        # Image saving is best-effort; do not affect core flow.
        pass


def append_sample(prompt: str, feat: np.ndarray, label: float, img: Any | None = None) -> int:
    """Append a labeled sample and optionally save its image.

    Minimal wrapper around append_dataset_row + save_sample_image to reduce
    call-site duplication. Returns the 1-based row index.
    """
    row_idx = append_dataset_row(prompt, feat, float(label))
    if img is not None:
        try:
            save_sample_image(prompt, row_idx, img)
        except Exception:
            pass
    return row_idx


# Legacy backup/aggregate NPZs removed (195c): folder-per-sample is the only format.


def dataset_stats_for_prompt(prompt: str) -> dict:
    """Folder-based stats: rows/pos/neg/d/recent_labels from data/<hash>/*/sample.npz."""
    rows = pos = neg = d = 0
    recent: list[int] = []
    root = data_root_for_prompt(prompt)
    if not os.path.isdir(root):
        return {"rows": 0, "pos": 0, "neg": 0, "d": 0, "recent_labels": recent}
    labels: list[int] = []
    for sample_path in _iter_sample_paths(root):
        Xi, yi = _load_sample_npz(sample_path)
        if Xi is None or yi is None:
            continue
        if d == 0 and getattr(Xi, "ndim", 1) == 2:
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
    # Import lazily so tests that stub latent_opt with minimal surface
    # (without dumps_state) can still import persistence.
    from ipo.core.latent_state import dumps_state  # local import

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
        if "app_version" in data.files:
            out["app_version"] = data["app_version"].item()
        if "created_at" in data.files:
            out["created_at"] = data["created_at"].item()
        if "prompt" in data.files:
            try:
                out["prompt"] = data["prompt"].item()
            except Exception:
                pass
    return out
_LOCKS_GUARD = threading.Lock()
_APPEND_LOCKS: dict[str, threading.Lock] = {}


def _lock_for_prompt(prompt: str) -> threading.Lock:
    """Return a stable mutex for this prompt's dataset folder.

    Keeps concurrency simple when many samples are saved quickly.
    """
    key = data_root_for_prompt(prompt)
    lock = _APPEND_LOCKS.get(key)
    if lock is None:
        with _LOCKS_GUARD:
            lock = _APPEND_LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                _APPEND_LOCKS[key] = lock
    return lock
_BASE_DIR_CACHE: str | None = None
