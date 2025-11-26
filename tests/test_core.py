import numpy as np
import pytest

def test_latent_state_init():
    from ipo.core.latent_state import init_latent_state
    state = init_latent_state(width=512, height=512)
    assert state.d == 4 * 64 * 64
    assert state.mu.shape == (state.d,)

def test_z_to_latents_not_zeros():
    from ipo.core.latent_state import init_latent_state, z_to_latents
    state = init_latent_state(width=512, height=512)
    z = np.random.randn(state.d)
    latents = z_to_latents(state, z)
    assert not np.allclose(latents, 0)

def test_z_from_prompt_deterministic():
    from ipo.core.latent_state import init_latent_state, z_from_prompt
    state = init_latent_state()
    z1 = z_from_prompt(state, "test")
    z2 = z_from_prompt(state, "test")
    assert np.allclose(z1, z2)

def test_ridge_fit():
    from ipo.core.latent_state import ridge_fit
    X = np.random.randn(10, 5)
    y = np.random.randn(10)
    w = ridge_fit(X, y, lam=1.0)
    assert w.shape == (5,)
    assert np.isfinite(w).all()

def test_value_model_fit():
    from ipo.core.latent_state import init_latent_state
    from ipo.core.value_model import fit_value_model
    state = init_latent_state()
    X = np.random.randn(10, state.d)
    y = np.array([1,-1,1,-1,1,-1,1,-1,1,-1])
    fit_value_model("Ridge", state, X, y, 1.0, {})
    assert not np.allclose(state.w, 0)
