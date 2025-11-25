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


def _append_row_and_fit(state: LatentState, diff: np.ndarray, label: np.ndarray, lam: float) -> None:
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
    d1 = _dir_w(state)
    # choose step size
    step = (
        (float(trust_r) / max(1, int(steps)))
        if (trust_r is not None and trust_r > 0)
        else (state.sigma / max(1, int(steps)))
    )
    if eta is not None:
        step = float(eta)
    # accumulate delta with optional trust clamp
    delta = _accumulate_delta(state.d, d1, int(max(1, int(steps))), float(step), trust_r)
    if gamma and float(gamma) != 0.0:
        d2 = _rand_orth_dir(state, d1)
        delta_plus = delta + float(gamma) * d2
        delta_minus = -delta - float(gamma) * d2
    else:
        delta_plus = delta
        delta_minus = -delta
    return z_p + _clamp_norm(delta_plus, trust_r), z_p + _clamp_norm(delta_minus, trust_r)


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
    d1 = _dir_w(state)
    S = float(trust_r) if (trust_r is not None and trust_r > 0) else float(state.sigma)
    mm = _linesearch_mags(mags, S, trust_r)
    # value is proportional to m for linear ridge; pick largest
    m_best = max(mm) if mm else 0.0
    delta = m_best * d1
    if gamma and float(gamma) != 0.0:
        d2 = _rand_orth_dir(state, d1)
        delta_plus = delta + float(gamma) * d2
        delta_minus = -delta - float(gamma) * d2
    else:
        delta_plus = delta
        delta_minus = -delta

    # final trust clamp
    return z_p + _clamp_norm(delta_plus, trust_r), z_p + _clamp_norm(
        delta_minus, trust_r
    )


def _accumulate_delta(d: int, d1: np.ndarray, steps: int, step: float, trust_r: Optional[float]) -> np.ndarray:
    """Accumulate steps along d1 with optional trust‑radius clamp (pure helper)."""
    delta = np.zeros(d, dtype=float)
    for _ in range(max(1, int(steps))):
        delta = delta + float(step) * d1
        if trust_r is not None and trust_r > 0:
            nn = float(np.linalg.norm(delta))
            if nn > trust_r and nn > 0.0:
                delta = delta * (trust_r / nn)
    return delta


def _rand_orth_dir(state: LatentState, d1: np.ndarray) -> np.ndarray:
    """Random unit direction orthogonal to d1 (pure helper)."""
    r = state.rng.standard_normal(state.d)
    d2 = _orth_component(r, d1)
    n = float(np.linalg.norm(d2))
    if n <= 1e-12:
        return _unit(d1)
    return d2 / n


def _linesearch_mags(mags: Optional[list[float]], S: float, trust_r: Optional[float]) -> list[float]:
    """Prepare candidate magnitudes for line-search with optional trust clamp."""
    cands = mags if (isinstance(mags, list) and len(mags) > 0) else [0.25 * S, 0.5 * S, 1.0 * S]
    out: list[float] = []
    for m in cands:
        m = float(max(0.0, m))
        if trust_r is not None and trust_r > 0 and m > float(trust_r):
            m = float(trust_r)
        out.append(m)
    return out


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
    yy = np.asarray(y, dtype=float).reshape(-1)
    L0, grad = _distance_loss_and_grad(mu, Z, yy, float(gamma))
    mu_new = mu - float(eta) * grad
    mu_new = _trust_clamp(mu_new, z_p, trust_r)
    try:
        L1 = _distance_loss_only(mu_new, Z, yy, float(gamma))
        if L0 is not None:
            print(f"[hill] L(mu) before={L0:.4f} after={L1:.4f}")
        else:
            print(f"[hill] L(mu) after={L1:.4f}")
    except Exception:
        pass
    state.mu = mu_new
    _push_mu_history(state)


def _distance_loss_and_grad(mu: np.ndarray, Z: np.ndarray, yy: np.ndarray, gamma: float) -> tuple[float | None, np.ndarray]:
    diffs = mu.reshape(1, -1) - Z  # shape (n, d)
    d2 = np.sum(diffs * diffs, axis=1)  # (n,)
    sig = _sigmoid(gamma * d2)
    try:
        L = float(np.sum(-yy * sig))
    except Exception:
        L = None
    scal = (yy) * sig * (1.0 - sig) * (2.0 * gamma)  # (n,)
    grad = (scal.reshape(-1, 1) * diffs).sum(axis=0)
    return L, grad


def _distance_loss_only(mu: np.ndarray, Z: np.ndarray, yy: np.ndarray, gamma: float) -> float:
    diffs = mu.reshape(1, -1) - Z
    d2 = np.sum(diffs * diffs, axis=1)
    sig = _sigmoid(gamma * d2)
    return float(np.sum(-yy * sig))


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
        mu, best_score = _best_of_along_d1(mu, z_p, d1, step_t, trust_r, scorer)
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
        mu, best_score = _best_of_along_d1(mu, z_p, d1, step_t, trust_r, _score)
        try:
            print(f"[xgb-hill-batch] step={t + 1} score={best_score:.4f}")
        except Exception:
            pass
    return mu


def _best_of_along_d1(mu: np.ndarray, z_p: np.ndarray, d1: np.ndarray, step: float, trust_r: Optional[float], scorer) -> tuple[np.ndarray, float]:
    """Evaluate ±step along d1 around mu and return (best_mu, best_score)."""
    candidates: list[tuple[float, np.ndarray]] = []
    for sgn in (1.0, -1.0):
        z_cand = mu + float(sgn) * float(step) * d1
        z_cand = _trust_clamp(z_cand, z_p, trust_r)
        delta = z_cand - z_p
        try:
            s = float(scorer(delta))
        except Exception:
            s = 0.0
        candidates.append((s, z_cand))
    try:
        best_score, best_z = max(candidates, key=lambda x: x[0])
    except ValueError:
        return mu, 0.0
    return best_z, float(best_score)


def _trust_clamp(z_cand: np.ndarray, z_p: np.ndarray, trust_r: Optional[float]) -> np.ndarray:
    if trust_r is None or float(trust_r) <= 0.0:
        return z_cand
    delta = z_cand - z_p
    r = float(np.linalg.norm(delta))
    if r > float(trust_r) and r > 0.0:
        return z_p + delta * (float(trust_r) / r)
    return z_cand




def _cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))




# Legacy non‑ridge proposers/scorers were removed to reduce LOC.
