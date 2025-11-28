import hashlib as _hashlib
import io
import os
import threading as _threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ipo.infra.constants import Config


@dataclass
class LatentState:
    width: int
    height: int
    d: int  # flattened length (latent or prompt embed)
    mu: np.ndarray
    sigma: float
    rng: np.random.Generator
    w: np.ndarray
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    step: int = 0
    z_pairs: Optional[np.ndarray] = None
    choices: Optional[np.ndarray] = None
    mu_hist: Optional[np.ndarray] = None
    space_mode: str = "Latent"  # "Latent" or "PromptEmbed"
    # Per-state lock for async updates to w
    w_lock: Any = field(default_factory=_threading.Lock, repr=False, compare=False)


def init_latent_state(
    width: int = Config.DEFAULT_WIDTH,
    height: int = Config.DEFAULT_HEIGHT,
    d: int = 0,
    seed: Optional[int] = 0,
    space_mode: str = "Latent",
) -> LatentState:
    h8, w8 = height // 8, width // 8
    rng = np.random.default_rng(seed)
    if space_mode == "PooledEmbed":
        from ipo.infra.pipeline_local import get_pooled_embed_dim
        d_eff = get_pooled_embed_dim() or 1024
    elif space_mode == "PromptEmbed":
        from ipo.infra.pipeline_local import get_prompt_embed_dim
        d_eff = get_prompt_embed_dim() or 59136
    else:
        d_eff = 4 * h8 * w8
    mu = np.zeros(d_eff, dtype=float)
    w = np.zeros(d_eff, dtype=float)
    return LatentState(width, height, d_eff, mu, 1.0, rng, w, space_mode=space_mode)


def save_state(state: LatentState, path: str) -> None:
    X = state.X if state.X is not None else np.zeros((0, state.d), dtype=float)
    y = state.y if state.y is not None else np.zeros((0,), dtype=float)
    z_pairs = getattr(state, "z_pairs", None)
    choices = getattr(state, "choices", None)
    mu_hist = getattr(state, "mu_hist", None)
    if z_pairs is None:
        z_pairs = np.zeros((0, 2, state.d), dtype=float)
    if choices is None:
        choices = np.zeros((0,), dtype=float)
    if mu_hist is None:
        mu_hist = np.zeros((0, state.d), dtype=float)
    np.savez_compressed(
        path,
        width=state.width,
        height=state.height,
        d=state.d,
        mu=state.mu,
        sigma=state.sigma,
        w=state.w,
        step=state.step,
        X=X,
        y=y,
        z_pairs=z_pairs,
        choices=choices,
        mu_hist=mu_hist,
    )
    print(f"[state] saved {path} d={state.d} {state.width}x{state.height} step={state.step}")


def _optional_arr(
    z: Mapping[str, Any], name: str, fallback_shape: tuple[int, ...]
) -> np.ndarray:
    arr = (
        z[name]
        if name in getattr(z, "files", z)
        else np.zeros(fallback_shape, dtype=float)
    )  # type: ignore[index]
    return arr


def _build_state_from_npz(z: Mapping[str, Any], seed: Optional[int]) -> LatentState:
    width = int(z["width"])  # type: ignore[index]
    height = int(z["height"])  # type: ignore[index]
    d = int(z["d"])  # type: ignore[index]
    mu = z["mu"]  # type: ignore[index]
    sigma = float(z["sigma"])  # type: ignore[index]
    w = z["w"]  # type: ignore[index]
    step = int(z["step"])  # type: ignore[index]
    X = _optional_arr(z, "X", (0, d))
    y = _optional_arr(z, "y", (0,))
    z_pairs = _optional_arr(z, "z_pairs", (0, 2, d))
    choices = _optional_arr(z, "choices", (0,))
    mu_hist = _optional_arr(z, "mu_hist", (0, d))
    rng = np.random.default_rng(seed)
    X_out = None if X.size == 0 else X
    y_out = None if y.size == 0 else y
    zp_out = None if z_pairs.size == 0 else z_pairs
    ch_out = None if choices.size == 0 else choices
    mh_out = None if mu_hist.size == 0 else mu_hist
    return LatentState(
        width, height, d, mu, sigma, rng, w, X_out, y_out, step, zp_out, ch_out, mh_out
    )


def load_state(path: str, seed: Optional[int] = 0) -> LatentState:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    z = np.load(path)
    return _build_state_from_npz(z, seed)


def dumps_state(state: LatentState) -> bytes:
    buf = io.BytesIO()
    X = state.X if state.X is not None else np.zeros((0, state.d), dtype=float)
    y = state.y if state.y is not None else np.zeros((0,), dtype=float)
    z_pairs = getattr(state, "z_pairs", None)
    choices = getattr(state, "choices", None)
    mu_hist = getattr(state, "mu_hist", None)
    if z_pairs is None:
        z_pairs = np.zeros((0, 2, state.d), dtype=float)
    if choices is None:
        choices = np.zeros((0,), dtype=float)
    if mu_hist is None:
        mu_hist = np.zeros((0, state.d), dtype=float)
    np.savez_compressed(
        buf,
        width=state.width,
        height=state.height,
        d=state.d,
        mu=state.mu,
        sigma=state.sigma,
        w=state.w,
        step=state.step,
        X=X,
        y=y,
        z_pairs=z_pairs,
        choices=choices,
        mu_hist=mu_hist,
    )
    return buf.getvalue()


def state_summary(state: LatentState) -> dict:
    mu_norm = float(np.linalg.norm(state.mu))
    w_norm = float(np.linalg.norm(state.w))
    z_pairs = getattr(state, "z_pairs", None)
    choices = getattr(state, "choices", None)
    X = getattr(state, "X", None)
    y = getattr(state, "y", None)
    return {
        "width": int(state.width),
        "height": int(state.height),
        "d": int(state.d),
        "step": int(state.step),
        "sigma": float(state.sigma),
        "mu_norm": mu_norm,
        "w_norm": w_norm,
        "pairs_logged": 0 if z_pairs is None else int(z_pairs.shape[0]),
        "choices_logged": 0 if choices is None else int(choices.shape[0]),
        "X_shape": None if X is None else tuple(X.shape),
        "y_len": None if y is None else int(len(y)),
    }


def loads_state(data: bytes, seed: Optional[int] = 0) -> LatentState:
    buf = io.BytesIO(data)
    z = np.load(buf)
    return _build_state_from_npz(z, seed)

# From deleted latent_logic.py

def ridge_fit(X, y, lam):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    if X.size == 0 or y.size == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 0)
    K = X @ X.T
    K.ravel()[::K.shape[1]+1] += float(lam)
    return X.T @ np.linalg.solve(K, y)

def z_to_latents(state, z, noise_gamma=0.0):
    h8, w8 = max(2, state.height//8), max(2, state.width//8)
    need = 4*h8*w8
    if z.size != need:
        z = np.resize(z, need)
    x = z.astype(np.float32).reshape(1, 4, h8, w8)
    if noise_gamma > 0:
        noise = state.rng.standard_normal(x.shape).astype(np.float32)
        x = noise_gamma * x + (1 - noise_gamma) * noise
    return x

def z_from_prompt(state, prompt):
    mode = getattr(state, "space_mode", "Latent")
    if mode == "PooledEmbed":
        # Return zeros - we optimize a delta added to base prompt
        return np.zeros(state.d, dtype=float)
    if mode == "PromptEmbed":
        from ipo.infra.pipeline_local import get_base_prompt_embed
        return get_base_prompt_embed(prompt).astype(float)
    h = int.from_bytes(_hashlib.sha1(prompt.encode()).digest()[:8], "big")
    rng = np.random.default_rng(h)
    return rng.standard_normal(state.d).astype(float) * state.sigma

def propose_pair_prompt_anchor(state, prompt, alpha=0.5, beta=0.5, trust_r=None):
    z_p = z_from_prompt(state, prompt)
    w = state.w[:state.d]
    n = float(np.linalg.norm(w))
    d1 = (w/n) if n > 1e-12 else state.rng.standard_normal(state.d)
    d1 = d1 / (float(np.linalg.norm(d1)) + 1e-12)
    return z_p + state.sigma*alpha*d1, z_p - state.sigma*beta*d1
