import unittest
import numpy as np
from latent_opt import init_latent_state, propose_latent_pair_ridge, z_to_latents, update_latent_ridge


class TestLatentOpt(unittest.TestCase):
    def test_shapes(self):
        st = init_latent_state(width=320, height=256, seed=0)
        z_a, z_b = propose_latent_pair_ridge(st)
        lat = z_to_latents(st, z_a)
        self.assertEqual(lat.shape, (1, 4, 256//8, 320//8))

    def test_update_moves_mean(self):
        st = init_latent_state(seed=1)
        z_a = np.ones(st.d)
        z_b = -np.ones(st.d)
        mu0 = st.mu.copy()
        update_latent_ridge(st, z_a, z_b, 'a', lr_mu=0.5)
        self.assertTrue(np.allclose(st.mu, mu0 + 0.5*(z_a - mu0)))

    def test_propose_pair_uses_w_direction(self):
        st = init_latent_state(seed=0)
        st.w[:] = 0.0
        st.w[0] = 1.0
        z1, z2 = propose_latent_pair_ridge(st, alpha=1.0, beta=1.0)
        # First proposal should move along +w direction
        self.assertGreater(z1[0], st.mu[0])

    def test_alpha_affects_step(self):
        st = init_latent_state(seed=0)
        st.w[:] = 0.0
        st.w[0] = 1.0
        z1a, _ = propose_latent_pair_ridge(st, alpha=1.0, beta=1.0)
        z1b, _ = propose_latent_pair_ridge(st, alpha=2.0, beta=1.0)
        self.assertGreater(z1b[0] - st.mu[0], z1a[0] - st.mu[0])


if __name__ == '__main__':
    unittest.main()
