from typing import Optional
import numpy as np
from latent_state import LatentState
from latent_ridge import append_pair, ridge_fit
import hashlib


def _clamp_norm(y: np.ndarray, r: Optional[float]) -> np.ndarray:
    if r is None:
        return y
    n = float(np.linalg.norm(y))
    if n <= r or n == 0.0:
        return y
    return y * (r / n)


# Logistic/epsilon-greedy proposal removed; ridge-only path retained.


def z_to_latents(state: LatentState, z: np.ndarray, noise_gamma: float = 0.35) -> np.ndarray:
    """Map flat z → latent tensor and blend in Gaussian noise.

    - Zero-centers per-channel means to avoid color bias.
    - Blends a small amount of Gaussian noise (state RNG) to prevent
      degenerate, low-rank latents that can decode to black.
    """
    h8, w8 = state.height // 8, state.width // 8
    x = z.astype(np.float32).reshape(1, 4, h8, w8)
    ch_mean = x.mean(axis=(0, 2, 3), keepdims=True)
    x = x - ch_mean
    if noise_gamma > 0.0:
        n = state.rng.standard_normal(x.shape).astype(np.float32)
        x = noise_gamma * x + (1.0 - noise_gamma) * n
    return x


# No logistic; only ridge update is supported.


# Logistic update removed; ridge update below is the only optimizer.


def update_latent_ridge(state: LatentState,
                        z_a: np.ndarray,
                        z_b: np.ndarray,
                        choice: str,
                        lr_mu: float = 0.3,
                        lam: float = 1e-2,
                        feats_a: Optional[np.ndarray] = None,
                        feats_b: Optional[np.ndarray] = None):
    if choice not in ('a', 'b'):
        raise ValueError("choice must be 'a' or 'b'")
    winner = z_a if choice == 'a' else z_b
    state.mu = state.mu + lr_mu * (winner - state.mu)
    if feats_a is not None and feats_b is not None:
        diff = (feats_a - feats_b).reshape(1, -1)
    else:
        diff = (z_a - z_b).reshape(1, -1)
    label = np.array([1.0 if choice == 'a' else -1.0])
    if state.X is None:
        state.X = diff
        state.y = label
    else:
        state.X = np.vstack([state.X, diff])
        state.y = np.concatenate([state.y, label]) if state.y is not None else label
    # Closed-form ridge: w = (X^T X + lam I)^{-1} X^T y
    if state.X is not None and state.X.shape[0] >= 1:
        state.w = ridge_fit(state.X, state.y, lam)  # type: ignore[arg-type]
    lbl = 1.0 if choice == 'a' else -1.0
    append_pair(state, z_a, z_b, lbl)
    state.sigma = max(0.2, state.sigma * 0.99)
    state.step += 1
    mu_now = state.mu.reshape(1, -1)
    mh = getattr(state, 'mu_hist', None)
    state.mu_hist = mu_now if mh is None else np.vstack([mh, mu_now])


def propose_latent_pair_ridge(state: LatentState, alpha: float = 0.5, beta: float = 0.5, trust_r: Optional[float] = None):
    w = state.w[: state.d]
    n = float(np.linalg.norm(w)) + 1e-12
    d1 = w / n
    idx_sorted = np.argsort(-np.abs(w))
    e = np.zeros_like(w)
    if len(idx_sorted) > 1:
        e[idx_sorted[1]] = 1.0
    else:
        e[0] = 1.0
    d2 = e - float(np.dot(e, d1)) * d1
    n2 = float(np.linalg.norm(d2))
    if n2 < 1e-12:
        r = state.rng.standard_normal(state.d)
        d2 = r - float(np.dot(r, d1)) * d1
        n2 = float(np.linalg.norm(d2)) + 1e-12
    d2 = d2 / n2
    z1 = state.mu + state.sigma * alpha * d1
    z2 = state.mu + state.sigma * beta * d2
    return _clamp_norm(z1, trust_r), _clamp_norm(z2, trust_r)


def z_from_prompt(state: LatentState, prompt: str) -> np.ndarray:
    """Deterministic z derived from the prompt text (hash-seeded Gaussian).

    Keeps things minimal: no model inversion, just a stable mapping prompt→z.
    """
    h = int.from_bytes(hashlib.sha1(prompt.encode('utf-8')).digest()[:8], 'big')
    rng = np.random.default_rng(h)
    z = rng.standard_normal(state.d).astype(float) * state.sigma
    return z


def propose_pair_prompt_anchor(state: LatentState,
                               prompt: str,
                               alpha: float = 0.5,
                               beta: float = 0.5,
                               trust_r: Optional[float] = None):
    """Propose a symmetric pair around z_prompt along the learned ridge direction.

    - Direction d1 comes from state.w; if degenerate, use a random direction.
    - Returns (z_plus, z_minus) = (z_p + σ·α·d1, z_p − σ·β·d1), optionally clamped
      so that ‖z − z_p‖ ≤ trust_r.
    """
    z_p = z_from_prompt(state, prompt)
    w = state.w[: state.d]
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        d1 = state.rng.standard_normal(state.d)
        n = float(np.linalg.norm(d1)) + 1e-12
    else:
        d1 = w
    d1 = d1 / n
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


def propose_pair_prompt_anchor_iterative(
    state: LatentState,
    prompt: str,
    steps: int = 3,
    eta: Optional[float] = None,
    trust_r: Optional[float] = None,
    gamma: float = 0.0,
):
    """Iteratively optimize Δ around z_prompt along w with tiny projected steps.

    - Starts at Δ=0 and takes `steps` projected steps along normalize(w).
    - Step length defaults to (trust_r/steps) if trust_r is set, otherwise σ/steps.
    - Returns symmetric pair (z_p+Δ, z_p−Δ).
    """
    z_p = z_from_prompt(state, prompt)
    w = state.w[: state.d]
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        d1 = state.rng.standard_normal(state.d)
        n = float(np.linalg.norm(d1)) + 1e-12
    else:
        d1 = w
    d1 = d1 / n
    step = (float(trust_r) / max(1, int(steps))) if (trust_r is not None and trust_r > 0) else (state.sigma / max(1, int(steps)))
    if eta is not None:
        step = float(eta)
    delta = np.zeros(state.d, dtype=float)
    for _ in range(max(1, int(steps))):
        delta = delta + step * d1
        if trust_r is not None and trust_r > 0:
            nn = float(np.linalg.norm(delta))
            if nn > trust_r and nn > 0.0:
                delta = delta * (trust_r / nn)
    if gamma and gamma != 0.0:
        r = state.rng.standard_normal(state.d)
        # make r orthogonal to d1
        r = r - float(np.dot(r, d1)) * d1
        nr = float(np.linalg.norm(r))
        d2 = (r / nr) if nr > 1e-12 else _clamp_norm(state.rng.standard_normal(state.d), 1.0)
        d2 = d2 / (float(np.linalg.norm(d2)) + 1e-12)
        delta_plus = delta + float(gamma) * d2
        delta_minus = -delta - float(gamma) * d2
    else:
        delta_plus = delta
        delta_minus = -delta
    def _cl(v):
        if trust_r is None or trust_r <= 0:
            return v
        n = float(np.linalg.norm(v))
        return v if (n <= trust_r or n == 0.0) else (v * (trust_r / n))
    return z_p + _cl(delta_plus), z_p + _cl(delta_minus)


def propose_pair_prompt_anchor_linesearch(
    state: LatentState,
    prompt: str,
    trust_r: Optional[float] = None,
    gamma: float = 0.0,
    mags: Optional[list[float]] = None,
):
    """Tiny line-search for Δ along w around z_prompt.

    - Candidates are magnitudes along d1 = normalize(w). If `mags` is None,
      use fractions of the available scale: [0.25, 0.5, 1.0] × S where
      S = trust_r (when provided) else state.sigma.
    - Choose the m that maximizes w·Δ (linear value), then return symmetric
      pair (z_p+Δ, z_p−Δ) with optional orthogonal γ·d2.
    """
    z_p = z_from_prompt(state, prompt)
    w = state.w[: state.d]
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        d1 = state.rng.standard_normal(state.d)
        n = float(np.linalg.norm(d1)) + 1e-12
    else:
        d1 = w
    d1 = d1 / n
    S = float(trust_r) if (trust_r is not None and trust_r > 0) else float(state.sigma)
    cands = mags if (isinstance(mags, list) and len(mags) > 0) else [0.25 * S, 0.5 * S, 1.0 * S]
    # clamp candidates to trust_r if needed
    mm = []
    for m in cands:
        m = float(max(0.0, m))
        if trust_r is not None and trust_r > 0 and m > float(trust_r):
            m = float(trust_r)
        mm.append(m)
    # value is proportional to m for linear ridge; pick largest
    m_best = max(mm) if mm else 0.0
    delta = m_best * d1
    if gamma and gamma != 0.0:
        r = state.rng.standard_normal(state.d)
        r = r - float(np.dot(r, d1)) * d1
        nr = float(np.linalg.norm(r))
        d2 = (r / nr) if nr > 1e-12 else d1
        d2 = d2 / (float(np.linalg.norm(d2)) + 1e-12)
        delta_plus = delta + float(gamma) * d2
        delta_minus = -delta - float(gamma) * d2
    else:
        delta_plus = delta
        delta_minus = -delta
    # final trust clamp
    def _cl(v):
        if trust_r is None or trust_r <= 0:
            return v
        n = float(np.linalg.norm(v))
        return v if (n <= trust_r or n == 0.0) else (v * (trust_r / n))
    return z_p + _cl(delta_plus), z_p + _cl(delta_minus)
