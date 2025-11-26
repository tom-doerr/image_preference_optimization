import hashlib, io, os, threading, fcntl
from datetime import datetime, timezone
import numpy as np
from ipo.infra.constants import APP_VERSION

def _hash(prompt): return hashlib.sha1(prompt.encode()).hexdigest()[:10]
def _base_dir(): return os.getenv("IPO_DATA_ROOT") or "data"
def data_root_for_prompt(prompt): return os.path.join(_base_dir(), _hash(prompt))


def state_path_for_prompt(prompt):
    root = data_root_for_prompt(prompt); os.makedirs(root, exist_ok=True)
    new_path = os.path.join(root, "latent_state.npz")
    legacy = f"latent_state_{_hash(prompt)}.npz"
    return legacy if os.path.exists(legacy) and not os.path.exists(new_path) else new_path


from contextlib import contextmanager
@contextmanager
def _file_lock(prompt):
    root = data_root_for_prompt(prompt); os.makedirs(root, exist_ok=True)
    f = open(os.path.join(root, ".lock"), "a+")
    try: fcntl.flock(f, fcntl.LOCK_EX); yield
    finally: fcntl.flock(f, fcntl.LOCK_UN); f.close()



def _target_dim(ss):
    lstate = getattr(ss, "lstate", None)
    return int(getattr(lstate, "d", 0) or 0) or None if lstate else None

def _iter_samples(root):
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n, "sample.npz")
        if os.path.exists(p): yield p

def _load_npz(path):
    with np.load(path) as d: return d.get("X"), d.get("y")


def get_dataset_for_prompt_or_session(prompt, ss):
    root = data_root_for_prompt(prompt)
    if not os.path.isdir(root):
        print(f"[data] no dir: {root}")
        return None, None
    target_d = _target_dim(ss)
    xs, ys = [], []
    for p in _iter_samples(root):
        Xi, yi = _load_npz(p)
        if Xi is None or yi is None: continue
        if target_d and Xi.shape[1] != target_d:
            print(f"[data] dim mismatch {Xi.shape[1]}!={target_d}")
            continue
        xs.append(Xi); ys.append(yi)
    if xs:
        print(f"[data] loaded {len(xs)} samples from {root}")
        return np.vstack(xs), np.concatenate(ys)
    print(f"[data] no samples in {root}")
    return None, None


_LOCKS = {}; _LOCKS_GUARD = threading.Lock()
def _mem_lock(prompt):
    key = data_root_for_prompt(prompt)
    with _LOCKS_GUARD:
        if key not in _LOCKS: _LOCKS[key] = threading.Lock()
    return _LOCKS[key]

def append_dataset_row(prompt, feat, label):
    import re
    root = data_root_for_prompt(prompt); os.makedirs(root, exist_ok=True)
    with _mem_lock(prompt), _file_lock(prompt):
        idxs = [int(m.group(1)) for n in os.listdir(root) if (m := re.fullmatch(r"(\d+)", n))]
        idx = (max(idxs) + 1) if idxs else 1
        while True:
            d = os.path.join(root, f"{idx:06d}")
            try: os.mkdir(d); break
            except FileExistsError: idx += 1
        path = os.path.join(d, "sample.npz")
        np.savez_compressed(path, X=feat, y=np.array([label]))
        print(f"[data] saved sample {idx} to {path}")
    return idx


def save_sample_image(prompt, row_idx, img):
    try:
        from PIL import Image
        d = os.path.join(data_root_for_prompt(prompt), f"{row_idx:06d}"); os.makedirs(d, exist_ok=True)
        with _mem_lock(prompt), _file_lock(prompt):
            (img if hasattr(img, "save") else Image.fromarray(np.asarray(img))).save(os.path.join(d, "image.png"))
    except: pass


def append_sample(prompt, feat, label, img=None):
    idx = append_dataset_row(prompt, feat, float(label))
    if img is not None: save_sample_image(prompt, idx, img)
    return idx

def export_state_bytes(state, prompt):
    from ipo.core.latent_state import dumps_state
    raw = dumps_state(state)
    with np.load(io.BytesIO(raw)) as d: items = {k: d[k] for k in d.files}
    items.update(prompt=np.array(prompt), created_at=np.array(datetime.now(timezone.utc).isoformat()), app_version=np.array(APP_VERSION))
    buf = io.BytesIO(); np.savez_compressed(buf, **items); return buf.getvalue()
