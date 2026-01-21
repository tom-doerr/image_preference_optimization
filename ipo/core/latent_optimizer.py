"""Latent optimization algorithms."""
import numpy as np

from ipo.infra.constants import DEFAULT_ITER_ETA, DEFAULT_XGB_OPTIM_MODE, Keys


def optimize_xgb(z, ls, ss, n, eta=DEFAULT_ITER_ETA):
    """Optimize z using XGBoost value model."""
    from ipo.core.optimizer import HillClimbOptimizer, LineSearchOptimizer, SPSAOptimizer
    from ipo.core.value_model import _get_xgb_model, _xgb_proba
    mdl = _get_xgb_model(ss)
    if mdl is None:
        return z
    max_r = float(ss.get(Keys.TRUST_R, 0) or 0)
    mode = ss.get(Keys.XGB_OPTIM_MODE) or DEFAULT_XGB_OPTIM_MODE
    def vf(x): return _xgb_proba(mdl, x)
    w = getattr(ls, "w", None)
    if mode == "Grad":
        mom = 0.9 if ss.get(Keys.XGB_MOMENTUM) else 0.0
        opt = SPSAOptimizer(eta=eta, momentum=mom, max_dist=max_r)
    elif mode == "Line" and w is not None and not np.allclose(w, 0):
        opt = LineSearchOptimizer(direction=w, eta=eta, max_dist=max_r)
    else:
        opt = HillClimbOptimizer(sigma=ls.sigma, eta=eta, max_dist=max_r)
    return opt.optimize(z, vf, n).z


def optimize_gauss(z, ls, ss, n, eta=DEFAULT_ITER_ETA):
    """Sample from fitted Gaussian, clamp per-dim deviation by Max Dist."""
    mu, sigma = ss.get("gauss_mu"), ss.get("gauss_sigma")
    if mu is None:
        return z
    rng = getattr(ls, "rng", None) or np.random.default_rng()
    delta = sigma * rng.standard_normal(len(mu))
    max_d = float(ss.get(Keys.TRUST_R, 0) or 0)
    if max_d > 0:
        delta = np.clip(delta, -max_d, max_d)
    return mu + delta


def optimize_latent(z, lstate, ss, steps, eta=DEFAULT_ITER_ETA):
    """Optimize z using selected value function."""
    if steps <= 0:
        return z
    vm = ss.get(Keys.VM_CHOICE) or "Ridge"
    if vm == "XGBoost":
        return optimize_xgb(z, lstate, ss, steps, eta)
    if vm == "Gaussian":
        return optimize_gauss(z, lstate, ss, steps, eta)
    from ipo.core.optimizer import RidgeOptimizer
    w = getattr(lstate, "w", None)
    if w is None or np.allclose(w, 0):
        return z
    max_r = float(ss.get(Keys.TRUST_R, 0) or 0)
    opt = RidgeOptimizer(w=w, eta=eta, max_dist=max_r)
    return opt.optimize(z, lambda x: float(np.dot(w, x)), steps).z
