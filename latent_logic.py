from typing import Optional
import hashlib
import numpy as np
from latent_state import LatentState


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
    if r is None:
        return y
    n = float(np.linalg.norm(y))
    if n <= r or n == 0.0:
        return y
    return y * (r / n)


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
    winner = z_a if choice == "a" else z_b
    state.mu = state.mu + lr_mu * (winner - state.mu)
    if feats_a is not None and feats_b is not None:
        diff = (feats_a - feats_b).reshape(1, -1)
    else:
        diff = (z_a - z_b).reshape(1, -1)
    label = np.array([1.0 if choice == "a" else -1.0])
    if state.X is None:
        state.X = diff
        state.y = label
    else:
        state.X = np.vstack([state.X, diff])
        state.y = np.concatenate([state.y, label]) if state.y is not None else label
    # Closed-form ridge: w = (X^T X + lam I)^{-1} X^T y
    if state.X is not None and state.X.shape[0] >= 1:
        state.w = ridge_fit(state.X, state.y, lam)  # type: ignore[arg-type]
    lbl = 1.0 if choice == "a" else -1.0
    append_pair(state, z_a, z_b, lbl)
    state.sigma = max(0.2, state.sigma * 0.99)
    state.step += 1
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
            print(
                f"[latent] z_from_prompt random anchor d={state.d} sigma={state.sigma:.3f} ‖z_p‖={n:.3f}"
            )
        except Exception:
            pass
        return z_cached
    h = int.from_bytes(hashlib.sha1(prompt.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(h)
    z = rng.standard_normal(state.d).astype(float) * state.sigma
    try:
        import numpy as _np  # local import

        n = float(_np.linalg.norm(z))
        print(
            f"[latent] z_from_prompt prompt_hash={h} d={state.d} sigma={state.sigma:.3f} ‖z_p‖={n:.3f}"
        )
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
    step = (
        (float(trust_r) / max(1, int(steps)))
        if (trust_r is not None and trust_r > 0)
        else (state.sigma / max(1, int(steps)))
    )
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
        d2 = (
            (r / nr)
            if nr > 1e-12
            else _clamp_norm(state.rng.standard_normal(state.d), 1.0)
        )
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
    cands = (
        mags
        if (isinstance(mags, list) and len(mags) > 0)
        else [0.25 * S, 0.5 * S, 1.0 * S]
    )
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


# propose_next_pair and ProposerOpts moved to proposer.py to centralize configuration


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def hill_climb_mu_distance(
    state: LatentState,
    prompt: str,
    X: np.ndarray,
    y: np.ndarray,
    eta: float = 0.2,
    gamma: float = 0.5,
    trust_r: Optional[float] = None,
) -> None:
    """One gradient step on μ to move toward positives and away from negatives.

    L(μ) = - Σ_i y_i · σ(γ · ||μ − z_i||^2), where z_i = z_prompt + X_i
    Step: μ ← μ − η · ∇L(μ) with optional trust‑radius clamp around z_prompt.
    """
    if X is None or y is None or len(getattr(y, "shape", (0,))) == 0:
        return
    z_p = z_from_prompt(state, prompt)
    mu = state.mu.astype(float)
    # If μ is still uninitialized (all zeros), start from a random point
    # around the prompt anchor to avoid always climbing from the same vector.
    if not np.any(mu):
        r = state.rng.standard_normal(state.d).astype(float)
        n = float(np.linalg.norm(r))
        if n > 0.0:
            r = r / n
        scale = (
            float(trust_r)
            if (trust_r is not None and float(trust_r) > 0.0)
            else float(state.sigma)
        )
        mu = z_p + scale * r
        state.mu = mu
    Z = z_p.reshape(1, -1) + np.asarray(X, dtype=float)
    diffs = mu.reshape(1, -1) - Z  # shape (n, d)
    d2 = np.sum(diffs * diffs, axis=1)  # (n,)
    yy = np.asarray(y, dtype=float).reshape(-1)
    sig = _sigmoid(gamma * d2)
    try:
        L0 = float(np.sum(-yy * sig))
    except Exception:
        L0 = None
    # scalers per sample
    scal = (yy) * sig * (1.0 - sig) * (2.0 * gamma)  # (n,)
    grad = (scal.reshape(-1, 1) * diffs).sum(axis=0)
    mu_new = mu - float(eta) * grad
    if trust_r is not None and float(trust_r) > 0.0:
        delta = mu_new - z_p
        n = float(np.linalg.norm(delta))
        if n > float(trust_r) and n > 0.0:
            mu_new = z_p + delta * (float(trust_r) / n)
    try:
        diffs_new = mu_new.reshape(1, -1) - Z
        d2_new = np.sum(diffs_new * diffs_new, axis=1)
        sig_new = _sigmoid(gamma * d2_new)
        L1 = float(np.sum(-yy * sig_new))
        if L0 is not None:
            print(f"[hill] L(mu) before={L0:.4f} after={L1:.4f}")
        else:
            print(f"[hill] L(mu) after={L1:.4f}")
    except Exception:
        pass
    state.mu = mu_new
    # record history
    mh = getattr(state, "mu_hist", None)
    mu_now = state.mu.reshape(1, -1)
    state.mu_hist = mu_now if mh is None else np.vstack([mh, mu_now])


def hill_climb_mu_xgb(
    state: LatentState,
    prompt: str,
    scorer,
    steps: int = 3,
    step_scale: float = 0.2,
    trust_r: Optional[float] = None,
) -> None:
    """Multi-step hill climb on μ using an external value scorer (e.g. XGBoost).

    Uses ridge w to define a direction d1 in latent space and, at each step,
    proposes μ ± step_t·d1, scores them via `scorer` on f = z − z_prompt,
    and moves μ to the better one. step_t decays as step_scale/(1+t).
    """
    try:
        n_steps = max(1, int(steps))
    except Exception:
        n_steps = 1
    if n_steps <= 0:
        return
    z_p = z_from_prompt(state, prompt)
    mu = state.mu.astype(float)
    if not np.any(mu):
        mu = z_p.copy()
    w = getattr(state, "w", None)
    if w is None:
        return
    w = np.asarray(w[: state.d], dtype=float)
    n = float(np.linalg.norm(w))
    if n == 0.0:
        return
    d1 = w / n
    try:
        base_step = float(step_scale)
    except Exception:
        base_step = 0.2
    for t in range(n_steps):
        step_t = base_step / float(1 + t)
        candidates = []
        for sgn in (1.0, -1.0):
            z_cand = mu + sgn * step_t * d1
            delta = z_cand - z_p
            if trust_r is not None and float(trust_r) > 0.0:
                r = float(np.linalg.norm(delta))
                if r > float(trust_r) and r > 0.0:
                    z_cand = z_p + delta * (float(trust_r) / r)
                    delta = z_cand - z_p
            fvec = delta
            try:
                score = float(scorer(fvec))
            except Exception:
                score = 0.0
            candidates.append((score, z_cand))
        try:
            best_score, best_z = max(candidates, key=lambda x: x[0])
        except ValueError:
            break
        mu = best_z
        try:
            print(f"[xgb-hill] step={t + 1} best_score={best_score:.4f}")
        except Exception:
            pass
    state.mu = mu
    mh = getattr(state, "mu_hist", None)
    mu_now = state.mu.reshape(1, -1)
    state.mu_hist = mu_now if mh is None else np.vstack([mh, mu_now])


def sample_z_xgb_hill(
    state: LatentState,
    prompt: str,
    scorer,
    steps: int = 3,
    step_scale: float = 0.2,
    trust_r: Optional[float] = None,
) -> np.ndarray:
    """Return one latent sample via XGB-guided hill climb from a random start.

    - Start from z_p + σ·r where r is a random unit vector.
    - Use ridge w to define d1 and perform a small multi-step best-of-two
      search along ±d1 with scores from `scorer(f)` where f = z − z_p.
    - If w is missing/zero or scorer fails, fall back to the random start.
    """
    z_p = z_from_prompt(state, prompt)
    # Random starting point around the anchor
    r = state.rng.standard_normal(state.d).astype(float)
    n_r = float(np.linalg.norm(r))
    if n_r > 0.0:
        r = r / n_r
    mu = z_p + float(state.sigma) * r

    w = getattr(state, "w", None)
    if w is None:
        try:
            print("[xgb-hill-batch] w is None; returning random start sample")
        except Exception:
            pass
        return mu
    w = np.asarray(w[: state.d], dtype=float)
    n_w = float(np.linalg.norm(w))
    if n_w == 0.0:
        try:
            print("[xgb-hill-batch] ‖w‖=0; returning random start sample")
        except Exception:
            pass
        return mu
    d1 = w / n_w
    try:
        n_steps = max(1, int(steps))
    except Exception:
        n_steps = 1
    try:
        base_step = float(step_scale)
    except Exception:
        base_step = 0.2

    # Fallback scorer when None: simple ridge dot
    def _score(delta: np.ndarray) -> float:
        if scorer is None:
            return float(np.dot(w, delta))
        try:
            return float(scorer(delta))
        except Exception:
            return float(np.dot(w, delta))

    for t in range(n_steps):
        step_t = base_step / float(1 + t)
        best_score = None
        best_z = None
        for sgn in (1.0, -1.0):
            z_cand = mu + sgn * step_t * d1
            delta = z_cand - z_p
            if trust_r is not None and float(trust_r) > 0.0:
                rdelta = float(np.linalg.norm(delta))
                if rdelta > float(trust_r) and rdelta > 0.0:
                    z_cand = z_p + delta * (float(trust_r) / rdelta)
                    delta = z_cand - z_p
            s = _score(delta)
            if best_score is None or s > best_score:
                best_score = s
                best_z = z_cand
        if best_z is None:
            break
        mu = best_z
        try:
            print(f"[xgb-hill-batch] step={t + 1} score={best_score:.4f}")
        except Exception:
            pass
    return mu


def propose_pair_distancehill(
    state: LatentState,
    prompt: str,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.5,
    gamma: float = 0.5,
    trust_r: Optional[float] = None,
):
    """Propose a symmetric pair around z_prompt along the negative gradient of L.

    Returns (z_plus, z_minus) = (z_p + σ·α·d1, z_p − σ·α·d1).
    """
    if X is None or y is None or len(getattr(y, "shape", (0,))) == 0:
        # fallback: small random directions
        r = state.rng.standard_normal(state.d)
        r = r / (float(np.linalg.norm(r)) + 1e-12)
        z_p = z_from_prompt(state, prompt)
        delta = state.sigma * float(alpha) * r
        return (z_p + delta, z_p - delta)
    z_p = z_from_prompt(state, prompt)
    mu = state.mu.astype(float)
    Z = z_p.reshape(1, -1) + np.asarray(X, dtype=float)
    diffs = mu.reshape(1, -1) - Z
    d2 = np.sum(diffs * diffs, axis=1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    sig = _sigmoid(gamma * d2)
    scal = (yy) * sig * (1.0 - sig) * (2.0 * gamma)
    grad = (scal.reshape(-1, 1) * diffs).sum(axis=0)
    n = float(np.linalg.norm(grad))
    if n == 0.0:
        r = state.rng.standard_normal(state.d)
        grad = r
        n = float(np.linalg.norm(grad)) + 1e-12
    d1 = grad / n
    delta = state.sigma * float(alpha) * d1

    def _clamp(z):
        if trust_r is None or float(trust_r) <= 0.0:
            return z
        d = z - z_p
        nn = float(np.linalg.norm(d))
        return (
            z
            if (nn <= float(trust_r) or nn == 0.0)
            else (z_p + d * (float(trust_r) / nn))
        )

    return _clamp(z_p + delta), _clamp(z_p - delta)


def distancehill_score(
    prompt: str,
    z_candidate: np.ndarray,
    state: LatentState,
    X: np.ndarray,
    y: np.ndarray,
    gamma: float = 0.5,
) -> float:
    """Return negative-activated distance objective for a single candidate.

    Smaller is better for a positive, larger for a negative; we return
    the signed sum as a scalar score (higher is better), so we negate L.
    """
    if X is None or y is None or len(getattr(y, "shape", (0,))) == 0:
        return 0.0
    z_p = z_from_prompt(state, prompt)
    Z = z_p.reshape(1, -1) + np.asarray(X, dtype=float)
    diffs = np.asarray(z_candidate, dtype=float).reshape(1, -1) - Z
    d2 = np.sum(diffs * diffs, axis=1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    sig = _sigmoid(gamma * d2)
    # Score = -L = +∑ y_i σ(γ d^2)
    return float(np.sum(yy * sig))


def _cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def hill_climb_mu_cosine(
    state: LatentState,
    prompt: str,
    X: np.ndarray,
    y: np.ndarray,
    eta: float = 0.2,
    beta: float = 5.0,
    trust_r: Optional[float] = None,
) -> None:
    """One gradient-ascent step on μ using logistic on cosine similarities.

    Maximize ∑ log σ(β y cos(μ−z_p, X_i)); step on μ with trust clamp.
    """
    if X is None or y is None or len(getattr(y, "shape", (0,))) == 0:
        return
    z_p = z_from_prompt(state, prompt)
    mu = state.mu.astype(float)
    mu_d = mu - z_p
    mu_n = float(np.linalg.norm(mu_d))
    eps = 1e-8
    if mu_n < eps:
        # kick a small random direction to avoid zero grad at origin
        r = state.rng.standard_normal(state.d)
        mu_d = r / (float(np.linalg.norm(r)) + eps) * 1e-3
        mu = z_p + mu_d
    grad = np.zeros_like(mu, dtype=float)
    mu_d = mu - z_p
    mu_n = float(np.linalg.norm(mu_d)) + eps
    mu_hat = mu_d / mu_n
    Y = np.asarray(y, dtype=float).reshape(-1)
    Xn = np.asarray(X, dtype=float)
    for xi, yi in zip(Xn, Y):
        xn = float(np.linalg.norm(xi))
        if xn < eps:
            continue
        x_hat = xi / xn
        s = float(np.dot(mu_hat, x_hat))  # cosine
        p = 1.0 / (1.0 + np.exp(-beta * yi * s))  # σ(β y s)
        # ∂s/∂μ = (x_hat - s μ_hat) / ||μ_d||
        ds = (x_hat - s * mu_hat) / mu_n
        g = (1.0 - p) * beta * yi * ds  # ascent grad on log σ
        grad += g
    mu_new = mu + float(eta) * grad
    if trust_r is not None and float(trust_r) > 0.0:
        d = mu_new - z_p
        n = float(np.linalg.norm(d))
        if n > float(trust_r) and n > 0.0:
            mu_new = z_p + d * (float(trust_r) / n)
    state.mu = mu_new
    mh = getattr(state, "mu_hist", None)
    mu_now = state.mu.reshape(1, -1)
    state.mu_hist = mu_now if mh is None else np.vstack([mh, mu_now])


def propose_pair_cosinehill(
    state: LatentState,
    prompt: str,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.5,
    beta: float = 5.0,
    trust_r: Optional[float] = None,
):
    """Propose symmetric pair around z_prompt along ascent direction of cosine logistic."""
    if X is None or y is None or len(getattr(y, "shape", (0,))) == 0:
        # random direction if no data
        r = state.rng.standard_normal(state.d)
        r = r / (float(np.linalg.norm(r)) + 1e-12)
        z_p = z_from_prompt(state, prompt)
        delta = state.sigma * float(alpha) * r
        return (z_p + delta, z_p - delta)
    # compute gradient at current μ and step direction
    z_p = z_from_prompt(state, prompt)
    mu = state.mu.astype(float)
    mu_d = mu - z_p
    mu_n = float(np.linalg.norm(mu_d))
    eps = 1e-8
    if mu_n < eps:
        mu_d = state.rng.standard_normal(state.d)
        mu_n = float(np.linalg.norm(mu_d)) + eps
    mu_hat = mu_d / mu_n
    grad = np.zeros_like(mu_d, dtype=float)
    Y = np.asarray(y, dtype=float).reshape(-1)
    Xn = np.asarray(X, dtype=float)
    for xi, yi in zip(Xn, Y):
        xn = float(np.linalg.norm(xi))
        if xn < eps:
            continue
        x_hat = xi / xn
        s = float(np.dot(mu_hat, x_hat))
        p = 1.0 / (1.0 + np.exp(-beta * yi * s))
        ds = (x_hat - s * mu_hat) / (mu_n + eps)
        grad += (1.0 - p) * beta * yi * ds
    n = float(np.linalg.norm(grad))
    if n < eps:
        r = state.rng.standard_normal(state.d)
        grad = r
        n = float(np.linalg.norm(grad)) + eps
    d1 = grad / n
    delta = state.sigma * float(alpha) * d1

    def _clamp(z):
        if trust_r is None or float(trust_r) <= 0.0:
            return z
        d = z - z_p
        nn = float(np.linalg.norm(d))
        return (
            z
            if (nn <= float(trust_r) or nn == 0.0)
            else (z_p + d * (float(trust_r) / nn))
        )

    return _clamp(z_p + delta), _clamp(z_p - delta)


def cosinehill_score(
    prompt: str,
    z_candidate: np.ndarray,
    state: LatentState,
    X: np.ndarray,
    y: np.ndarray,
    beta: float = 5.0,
) -> float:
    if X is None or y is None or len(getattr(y, "shape", (0,))) == 0:
        return 0.0
    z_p = z_from_prompt(state, prompt)
    mu_d = np.asarray(z_candidate, dtype=float) - z_p
    mu_n = float(np.linalg.norm(mu_d))
    if mu_n < 1e-8:
        return 0.0
    mu_hat = mu_d / mu_n
    Y = np.asarray(y, dtype=float).reshape(-1)
    Xn = np.asarray(X, dtype=float)
    out = 0.0
    for xi, yi in zip(Xn, Y):
        xn = float(np.linalg.norm(xi))
        if xn < 1e-8:
            continue
        x_hat = xi / xn
        s = float(np.dot(mu_hat, x_hat))
        out += yi * (1.0 / (1.0 + np.exp(-beta * s)))
    return float(out)
