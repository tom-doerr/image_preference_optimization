import hashlib
from typing import Optional

import numpy as np

from ipo.core.latent_state import LatentState


def append_pair(
    state: LatentState, z_a: np.ndarray, z_b: np.ndarray, label: float
) -> None:
    pair = np.stack([z_a.astype(float), z_b.astype(float)], axis=0).reshape(
        1, 2, state.d
    )
    zp = getattr(state, "z_pairs", None)
    ch = getattr(state, "choices", None)
    state.z_pairs = pair if zp is None else np.vstack([zp, pair])
    state.choices = np.array([label]) if ch is None else np.concatenate([ch, [label]])


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge using the dual to avoid a d×d solve.

    w = X^T (XX^T + λI)^{-1} y
    This is much faster/stabler when feature dim d ≫ rows n (our case).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.size == 0 or y.size == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 0, dtype=float)
    K = X @ X.T
    # add λ to the diagonal in-place (works for any square K)
    K.ravel()[:: K.shape[1] + 1] += float(lam)
    alpha = np.linalg.solve(K, y)
    return X.T @ alpha


def _clamp_norm(y: np.ndarray, r: Optional[float]) -> np.ndarray:
    """Clamp vector norm to r (no-op when r is falsy)."""
    if r is None or r <= 0:
        return y
    n = float(np.linalg.norm(y))
    return y if (n == 0.0 or n <= float(r)) else (y * (float(r) / n))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def _dir_w(state: LatentState) -> np.ndarray:
    """Unit direction from ridge weights or a random fallback."""
    w = np.asarray(state.w[: state.d], dtype=float)
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        r = state.rng.standard_normal(state.d)
        return _unit(r)
    return w / n


def _orth_component(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return component of v orthogonal to unit u."""
    return v - float(np.dot(v, u)) * u


# Logistic/epsilon-greedy proposal removed; ridge-only path retained.


def z_to_latents(
    state: LatentState, z: np.ndarray, noise_gamma: float = 0.35
) -> np.ndarray:
    """Map flat z → latent tensor and blend in Gaussian noise.

    - Zero-centers per-channel means to avoid color bias.
    - Blends a small amount of Gaussian noise (state RNG) to prevent
      degenerate, low-rank latents that can decode to black.
    """
    h8, w8 = state.height // 8, state.width // 8
    # Guard tiny test sizes: enforce minimum latent grid 2×2 to avoid reshape errors
    if h8 < 2:
        h8 = 2
    if w8 < 2:
        w8 = 2
    # If z length doesn't match 4*h8*w8 (e.g., after a stub size change), reinit a tiny z
    need = 4 * h8 * w8
    if z.size != need:
        z = np.zeros(need, dtype=np.float32)
    x = z.astype(np.float32).reshape(1, 4, h8, w8)
    ch_mean = x.mean(axis=(0, 2, 3), keepdims=True)
    x = x - ch_mean
    if noise_gamma > 0.0:
        n = state.rng.standard_normal(x.shape).astype(np.float32)
        x = noise_gamma * x + (1.0 - noise_gamma) * n
        # Re-center per-channel after blending to keep zero-mean invariant
        ch_mean2 = x.mean(axis=(0, 2, 3), keepdims=True)
        x = x - ch_mean2
    return x


# No logistic; only ridge update is supported.


# Logistic update removed; ridge update below is the only optimizer.


def update_latent_ridge(
    state: LatentState,
    z_a: np.ndarray,
    z_b: np.ndarray,
    choice: str,
    lr_mu: float = 0.3,
    lam: float = 1e-3,
    feats_a: Optional[np.ndarray] = None,
    feats_b: Optional[np.ndarray] = None,
):
    if choice not in ("a", "b"):
        raise ValueError("choice must be 'a' or 'b'")
    winner = _winner_vector(choice, z_a, z_b)
    _update_mu_inplace(state, winner, lr_mu)
    diff = _feature_diff(z_a, z_b, feats_a, feats_b)
    label = np.array([1.0 if choice == "a" else -1.0])
    _append_row_and_fit(state, diff, label, lam)
    append_pair(state, z_a, z_b, float(label[0]))
    state.sigma = max(0.2, state.sigma * 0.99)
    state.step += 1
    _push_mu_history(state)


def _winner_vector(choice: str, z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    return z_a if choice == "a" else z_b


def _feature_diff(z_a: np.ndarray, z_b: np.ndarray, feats_a, feats_b) -> np.ndarray:
    if feats_a is not None and feats_b is not None:
        return (feats_a - feats_b).reshape(1, -1)
    return (z_a - z_b).reshape(1, -1)


def _append_row_and_fit(state: LatentState, diff: np.ndarray, label: np.ndarray, lam: float) -> None:  # noqa: E501
    if state.X is None:
        state.X = diff
        state.y = label
    else:
        state.X = np.vstack([state.X, diff])
        state.y = np.concatenate([state.y, label]) if state.y is not None else label
    if state.X is not None and state.X.shape[0] >= 1:
        state.w = ridge_fit(state.X, state.y, lam)  # type: ignore[arg-type]


def _update_mu_inplace(state: LatentState, winner: np.ndarray, lr_mu: float) -> None:
    state.mu = state.mu + float(lr_mu) * (winner - state.mu)


def _push_mu_history(state: LatentState) -> None:
    mu_now = state.mu.reshape(1, -1)
    mh = getattr(state, "mu_hist", None)
    state.mu_hist = mu_now if mh is None else np.vstack([mh, mu_now])


def propose_latent_pair_ridge(
    state: LatentState,
    alpha: float = 0.5,
    beta: float = 0.5,
    trust_r: Optional[float] = None,
):
    w = state.w[: state.d]
    d1 = _unit(w)
    idx_sorted = np.argsort(-np.abs(w))
    e = np.zeros_like(w)
    if len(idx_sorted) > 1:
        e[idx_sorted[1]] = 1.0
    else:
        e[0] = 1.0
    d2 = _orth_component(e, d1)
    if float(np.linalg.norm(d2)) < 1e-12:
        r = state.rng.standard_normal(state.d)
        d2 = _orth_component(r, d1)
    d2 = _unit(d2)
    z1 = state.mu + state.sigma * alpha * d1
    z2 = state.mu + state.sigma * beta * d2
    return _clamp_norm(z1, trust_r), _clamp_norm(z2, trust_r)


def z_from_prompt(state: LatentState, prompt: str) -> np.ndarray:
    """Latent anchor z for a prompt or a random anchor when enabled.

    Default: deterministic hash-seeded Gaussian from the prompt text.
    When `state.use_random_anchor` is truthy, use a single random anchor
    per state (cached on `state.random_anchor_z`) instead.
    """
    # Optional random-anchor mode (set from the UI via state attribute)
    if getattr(state, "use_random_anchor", False):
        z_cached = getattr(state, "random_anchor_z", None)
        if z_cached is None or getattr(z_cached, "shape", (0,))[0] != state.d:
            rng = getattr(state, "rng", np.random.default_rng())
            z_cached = rng.standard_normal(state.d).astype(float) * state.sigma
            state.random_anchor_z = z_cached
        try:
            import numpy as _np  # local to keep deps minimal

            n = float(_np.linalg.norm(z_cached))
            print(f"[latent] z_from_prompt random anchor d={state.d} sigma={state.sigma:.3f} ‖z_p‖={n:.3f}")  # noqa: E501
        except Exception:
            pass
        return z_cached
    h = int.from_bytes(hashlib.sha1(prompt.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(h)
    z = rng.standard_normal(state.d).astype(float) * state.sigma
    try:
        import numpy as _np  # local import

        n = float(_np.linalg.norm(z))
        print(f"[latent] z_from_prompt prompt_hash={h} d={state.d} sigma={state.sigma:.3f} ‖z_p‖={n:.3f}")  # noqa: E501
    except Exception:
        pass
    return z


def propose_pair_prompt_anchor(
    state: LatentState,
    prompt: str,
    alpha: float = 0.5,
    beta: float = 0.5,
    trust_r: Optional[float] = None,
):
    """Propose a symmetric pair around z_prompt along the learned ridge direction.

    - Direction d1 comes from state.w; if degenerate, use a random direction.
    - Returns (z_plus, z_minus) = (z_p + σ·α·d1, z_p − σ·β·d1), optionally clamped
      so that ‖z − z_p‖ ≤ trust_r.
    """
    z_p = z_from_prompt(state, prompt)
    d1 = _dir_w(state)
    d_plus = state.sigma * alpha * d1
    d_minus = -state.sigma * beta * d1
    if trust_r is not None and trust_r > 0:

        def _clamp_delta(d):
            nn = float(np.linalg.norm(d))
            if nn <= trust_r or nn == 0.0:
                return d
            return d * (trust_r / nn)

        d_plus = _clamp_delta(d_plus)
        d_minus = _clamp_delta(d_minus)
    return z_p + d_plus, z_p + d_minus






























